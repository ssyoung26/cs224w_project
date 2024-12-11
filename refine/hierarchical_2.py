import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from structgen.encoder import MPNEncoder
from structgen.data import alphabet
from structgen.utils import *
from structgen.protein_features import ProteinFeatures
from torch.utils.checkpoint import checkpoint
# Ignore these if torch geometric is not in your environment
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv


class HierarchicalEncoder2(nn.Module):
    
    def __init__(self, args, node_in, edge_in):
        super(HierarchicalEncoder2, self).__init__()
        self.node_in, self.edge_in = node_in, edge_in
        self.W_v = nn.Sequential(
                nn.Linear(self.node_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
                nn.Linear(self.edge_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.depth)
        ])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, V, E, hS, E_idx, mask):
        hS = hS.to(V.dtype)
        mask = mask.to(V.dtype)

        h_v = self.W_v(V)  # [B, N, H] 
        h_e = self.W_e(E)  # [B, N, K, H] 
        nei_s = gather_nodes(hS, E_idx)  # [B, N, K, H]

        # [B, N, 1] -> [B, N, K, 1] -> [B, N, K]
        vmask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        h = h_v
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)  # [B, N, K, H]
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            # h = layer(h, nei_h, mask_attend=vmask)  # [B, N, H]
            h = checkpoint(layer, h, nei_h, vmask)

            h = h * mask.unsqueeze(-1)  # [B, N, H]
        return h


class HierarchicalDecoder2(nn.Module):

    def __init__(self, args):
        super(HierarchicalDecoder2, self).__init__()
        self.cdr_type = args.cdr_type
        self.k_neighbors = args.k_neighbors
        self.block_size = args.block_size
        self.update_freq = args.update_freq
        self.hidden_size = args.hidden_size
        self.pos_embedding = PosEmbedding(16)
        
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        self.O_d0 = nn.Linear(args.hidden_size, 12)
        self.O_d = nn.Linear(args.hidden_size, 12)
        self.O_s = nn.Linear(args.hidden_size, args.vocab_size)
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)

        self.struct_mpn = HierarchicalEncoder2(args, self.node_in, self.edge_in)
        self.seq_mpn = HierarchicalEncoder2(args, self.node_in, self.edge_in)
        self.init_mpn = HierarchicalEncoder2(args, 16, 32)
        self.rnn = nn.GRU(
                args.hidden_size, args.hidden_size, batch_first=True, 
                num_layers=1, bidirectional=True
        ) 
        self.W_stc = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )
        self.W_seq = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        # Changed this for Loss function test
        self.mse_loss = nn.MSELoss(reduction='none')
        # self.mse_loss = nn.L1Loss(reduction='none')

        self.diffusion_model = DiffusionModel(hidden_dim=args.hidden_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def init_struct(self, B, N, K):
        # initial V
        pos = torch.arange(N).cuda()
        V = self.pos_embedding(pos.view(1, N, 1))  # [1, N, 1, 16]
        V = V.squeeze(2).expand(B, -1, -1)  # [B, N, 6]
        # initial E_idx
        pos = pos.unsqueeze(0) - pos.unsqueeze(1)     # [N, N]
        D_idx, E_idx = pos.abs().topk(k=K, dim=-1, largest=False)    # [N, K]
        E_idx = E_idx.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]
        D_idx = D_idx.unsqueeze(0).expand(B, -1, -1)  # [B, N, K]
        # initial E
        E_rbf = self.features._rbf(3 * D_idx)
        E_pos = self.features.embeddings(E_idx)
        E = torch.cat((E_pos, E_rbf), dim=-1)
        return V, E, E_idx

    def init_coords(self, S, mask):
        B, N = S.size(0), S.size(1)
        K = min(self.k_neighbors, N)
        V, E, E_idx = self.init_struct(B, N, K)
        
        V = V.float()
        E = E.float()
        S = S.float()
        mask = mask.float()

        h = self.init_mpn(V, E, S, E_idx, mask)
        return self.predict_dist(self.O_d0(h))

    # Q: [B, N, H], K, V: [B, M, H]
    def attention(self, Q, context, cmask, W):
        att = torch.bmm(Q, context.transpose(1, 2))  # [B, N, M]
        att = att - 1e6 * (1 - cmask.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, context)  # [B, N, M] * [B, M, H]
        out = torch.cat([Q, out], dim=-1)
        return W(out)

    def predict_dist(self, X):
        X = X.view(X.size(0), X.size(1), 4, 3)
        X_ca = X[:, :, 1, :]
        dX = X_ca[:, None, :, :] - X_ca[:, :, None, :]
        D = torch.sum(dX ** 2, dim=-1)
        V = self.features._dihedrals(X)
        AD = self.features._AD_features(X[:,:,1,:])
        return X.detach().clone(), D, V, AD

    def mask_mean(self, X, mask, i):
        # [B, N, 4, 3] -> [B, 1, 4, 3] / [B, 1, 1, 1]
        X = X[:, i:i+self.block_size]
        if X.dim() == 4:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1).unsqueeze(-1)
        else:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1)
        return torch.sum(X * mask, dim=1, keepdims=True) / (mask.sum(dim=1, keepdims=True) + 1e-8)

    def make_X_blocks(self, X, l, r, mask):
        N = X.size(1)
        lblocks = [self.mask_mean(X, mask, i) for i in range(0, l, self.block_size)]
        rblocks = [self.mask_mean(X, mask, i) for i in range(r + 1, N, self.block_size)]
        bX = torch.cat(lblocks + [X[:, l:r+1]] + rblocks, dim=1)
        return bX.detach()

    def make_S_blocks(self, LS, S, RS, l, r, mask):
        N = S.size(1)
        hS = self.W_s(S.long())
        LS = [self.mask_mean(hS, mask, i) for i in range(0, l, self.block_size)]
        RS = [self.mask_mean(hS, mask, i) for i in range(r + 1, N, self.block_size)]
        bS = torch.cat(LS + [hS[:, l:r+1]] + RS, dim=1)
        lmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(0, l, self.block_size)]
        rmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(r + 1, N, self.block_size)]
        bmask = torch.cat(lmask + [mask[:, l:r+1]] + rmask, dim=1)
        return bS, bmask, len(LS), len(RS)

    def get_completion_mask(self, B, N, cdr_range):
        cmask = torch.zeros(B, N).cuda()
        for i, (l,r) in enumerate(cdr_range):
            cmask[i, l:r+1] = 1
        return cmask

    def remove_cdr_coords(self, X, cdr_range):
        X = X.clone()
        for i, (l,r) in enumerate(cdr_range):
            X[i, l:r+1, :, :] = 0
        return X.clone()

    def forward(self, true_X, true_S, true_cdr, mask, antigen_coords=None, antigen_seq=None):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        # Ensure dtype consistency within mixed precision
        with torch.cuda.amp.autocast():
            cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
            T_min = min([l for l, r in cdr_range])
            T_max = max([r for l, r in cdr_range])
            cmask = self.get_completion_mask(B, N, cdr_range)
            smask = mask.clone()

            # Encode framework
            S = true_S.clone() * (1 - cmask.long())
            hS, _ = self.rnn(self.W_s(S.long()))
            LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
            hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
            cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

            # Ground truth
            true_X = self.make_X_blocks(true_X, T_min, T_max, smask)
            true_V = self.features._dihedrals(true_X)
            true_AD = self.features._AD_features(true_X[:, :, 1, :])
            true_D, mask_2D = pairwise_distance(true_X, mask)
            true_D = true_D ** 2

            # Initialize
            sloss = 0.0
            X, D, V, AD = self.init_coords(hS, mask)
            X = X.detach().clone()
            dloss = self.huber_loss(D, true_D)
            vloss = self.mse_loss(V, true_V)
            aloss = self.mse_loss(AD, true_AD)

            # Toggle this to not include extra antigen_constraint loss term
            antigen_coords = None
            if antigen_coords is not None:
                # For just the naive approach:
                # antigen_constraint_loss = self.apply_antigen_constraint(predicted_ca=X[:, :, 1, :], true_ca=true_X[:, :, 1, :], antigen_coords=antigen_coords)
                
                # For diffusion model
                antigen_constraint_loss = self.apply_antigen_constraint_diffusion(predicted_ca=X[:, :, 1, :], true_ca=true_X[:, :, 1, :], antigen_coords=antigen_coords)
            else:
                antigen_constraint_loss = 0.0

            for t in range(T_min, T_max + 1):
                # Prepare input
                V, E, E_idx = self.features(X, mask)
                hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

                # Predict residue t
                h = self.seq_mpn(V, E, hS, E_idx, mask)
                h = self.attention(h, LS, smask, self.W_seq)
                logits = self.O_s(h[:, offset + t - T_min])
                logits = logits.float()
                snll = self.ce_loss(logits, true_S[:, t].long())
                sloss = sloss + torch.sum(snll * cmask[:, offset + t - T_min])

                # Teacher forcing on S
                S = S.clone()
                S[:, t] = true_S[:, t]

                # Iterative refinement
                if t % self.update_freq == 0:
                    h = self.struct_mpn(V, E, hS, E_idx, mask)
                    h = self.attention(h, LS, smask, self.W_stc)
                    X, D, V, AD = self.predict_dist(self.O_d(h))
                    X = X.detach().clone()
                    dloss = dloss + self.huber_loss(D, true_D)
                    vloss = vloss + self.mse_loss(V, true_V)
                    aloss = aloss + self.mse_loss(AD, true_AD)

            dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
            vloss = torch.sum(vloss * mask.unsqueeze(-1)) / mask.sum()
            aloss = torch.sum(aloss * mask.unsqueeze(-1)) / mask.sum()
            sloss = sloss.sum() / cmask.sum()
            loss = sloss + dloss + vloss + aloss + 10*antigen_constraint_loss
            return loss, sloss

    def log_prob(self, true_S, true_cdr, mask, antigen_coords=None, antigen_seq=None):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l, r in cdr_range])
        T_max = max([r for l, r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # Initialize
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        # Placeholder for antigen-based adjustments (if required)
        if antigen_coords is not None and antigen_seq is not None:
            # Optionally process antigen_coords and antigen_seq here
            pass

        sloss = 0.0
        X = self.init_coords(hS, mask)[0]
        X = X.detach().clone()

        for t in range(T_min, T_max + 1):
            # Prepare input
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # Predict residue t
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_seq)
            logits = self.O_s(h[:, offset + t - T_min])
            logits = logits.float()
            snll = self.ce_loss(logits, true_S[:, t].long())
            sloss = sloss + snll * cmask[:, offset + t - T_min]

            # Teacher forcing on S
            S = S.clone()
            S[:, t] = true_S[:, t]

            # Iterative refinement
            if t % self.update_freq == 0:
                h = self.struct_mpn(V, E, hS, E_idx, mask)
                h = self.attention(h, LS, smask, self.W_stc)
                X = self.predict_dist(self.O_d(h))[0]
                X = X.detach().clone()

        ppl = sloss / cmask.sum(dim=-1)
        sloss = sloss.sum() / cmask.sum()
        return ReturnType(nll=sloss, ppl=ppl, X=X, X_cdr=X[:, offset:offset+T_max-T_min+1])

        
    def generate(self, true_S, true_cdr, mask, return_ppl=False):
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # initialize
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        X = self.init_coords(hS, mask)[0]
        X = X.detach().clone()
        sloss = 0

        for t in range(T_min, T_max + 1):
            # Prepare input
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # Predict residue t
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_seq)
            logits = self.O_s(h[:, offset + t - T_min])
            prob = F.softmax(logits, dim=-1)  # [B, 20]
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)  # [B, 1]
            sloss = sloss + self.ce_loss(logits, S[:, t]) * cmask[:, offset + t - T_min]

            # Iterative refinement
            h = self.struct_mpn(V, E, hS, E_idx, mask)
            h = self.attention(h, LS, smask, self.W_stc)
            X = self.predict_dist(self.O_d(h))[0]
            X = X.detach().clone()

        S = S.tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(cdr_range[i][0], cdr_range[i][1] + 1)]) for i in range(B)]
        ppl = torch.exp(sloss / cmask.sum(dim=-1))
        return (S, ppl, X[:, offset:offset+T_max-T_min+1]) if return_ppl else S

    def generate_inverse_probability_map(self, antibody_coords, antigen_coords=None):
        """
        New function: Generate an inverse probability map based on distances.
        """
        if antigen_coords is not None:
            # Reduce antigen_coords along the third dimension (e.g., average over atoms)
            antigen_coords_reduced = antigen_coords.mean(dim=2)  # Shape becomes [B, 5, 3]
            # Compute pairwise distances
            dX = antibody_coords.unsqueeze(2) - antigen_coords_reduced.unsqueeze(1)  # Shape: [B, 32, 5, 3]
            distances = torch.norm(dX, dim=-1)  # Shape: [B, 32, 5]
            # Apply a Gaussian kernel to derive an inverse probability map
            sigma = 5.0
            inv_prob_map = torch.exp(-distances**2 / (2 * sigma**2))
            inv_prob_map = 1 - inv_prob_map
            return inv_prob_map
        else:
            return None

    def apply_antigen_constraint(self, predicted_ca, true_ca=None, antigen_coords=None):
        """
        New function: Apply antigen-based constraints as a loss term.
        """
        # print(f"predicted_ca shape: {predicted_ca.shape}")
        # if true_ca is not None:
        #     # print(f"true_ca shape: {true_ca.shape}")
        #     print("")
        # if antigen_coords is not None:
        #     # print(f"antigen_coords shape: {antigen_coords.shape}")
        #     print("")

        # Generate inverse probability map
        inv_prob_map = self.generate_inverse_probability_map(predicted_ca, antigen_coords)

        if inv_prob_map is not None:
            # Compute pairwise distances between predicted and true antibody C-alpha coordinates
            dX = predicted_ca - true_ca
            distances = torch.norm(dX, dim=-1)  # [B, N]

            # Apply the inverse probability map as a weight then aggregate distances
            weighted_distances = distances * inv_prob_map.mean(dim=-1)  # [B, N]
            antigen_constraint_loss = weighted_distances.mean()
            # print(f"antigen_constraint_loss: {antigen_constraint_loss}")

            return antigen_constraint_loss
        else:
            # print("inv_prob_map is None")
            return 0.0

    def generate_inverse_probability_map_diffusion(self, antibody_coords, antigen_coords=None):
        """
        New function: Generate an inverse probability map using a diffusion model.
        """
        if antigen_coords is not None:
            # Pool antigen atomic coordinates (e.g., mean along atom dimension)
            pooled_antigen_coords = antigen_coords.mean(dim=2)  # [B, N_antigen, 3]
            combined_coords = torch.cat([antibody_coords, pooled_antigen_coords], dim=1)  # [B, N_combined, 3]

            # Generate initial noisy map (Gaussian noise)
            B, N_combined, _ = combined_coords.size()
            noisy_map = torch.randn(B, N_combined, 3).to(combined_coords.device).view(-1, 3)  # [total_nodes, 3]

            edge_index, batch = self.construct_graph_edges(combined_coords, B)
            timesteps = torch.randint(0, 1000, (B,), device=combined_coords.device)  # [batch_size]
            inv_prob_map = self.diffusion_model(noisy_map, timesteps, edge_index, batch)

            return inv_prob_map.view(B, N_combined, -1)  # Reshape back to [B, N_combined, features]
        else:
            return None

    def apply_antigen_constraint_diffusion(self, predicted_ca, true_ca=None, antigen_coords=None, epoch=0, max_epochs=100):
        if antigen_coords is not None:
            # Generate inverse probability map
            inv_prob_map = self.generate_inverse_probability_map_diffusion(predicted_ca, antigen_coords)
            if inv_prob_map is not None:
                # Normalize probabilities
                inv_prob_map = F.softmax(inv_prob_map, dim=1)
                dX = predicted_ca - true_ca
                distances = torch.norm(dX, dim=-1)  # [B, N]
                # Slice probabilities for antibody nodes
                inv_prob_map_antibody = inv_prob_map[:, :predicted_ca.size(1)]
                weighted_distances = distances * inv_prob_map_antibody.mean(dim=-1)

                # With regularization:
                # regularization_weight = max(0.1, 0.01 * (1 - epoch / max_epochs))
                # antigen_constraint_loss = weighted_distances.mean() + regularization_weight * torch.mean((1 - inv_prob_map_antibody) ** 2)

                # Without regularization: 
                antigen_constraint_loss = weighted_distances.mean()
                # print(f" antigen_constraint_loss: {antigen_constraint_loss}")

                return antigen_constraint_loss
            else:
                return 0.0
        else:
            return 0.0
            
    def construct_graph_edges(self, combined_coords, batch_size):
        """
        New function:
            Construct edges (graph will be a fully connected graph) 
            between Ab/Ag and batch indices for the graph 
            representation.

        Args:
            combined_coords: Combined antibody and antigen coordinates [B, N_combined, 3].
            batch_size: Number of graphs in the batch.

        Returns:
            edge_index: Edge list [2, total_edges].
            batch: Batch indices [total_nodes].
        """
        B, N, _ = combined_coords.size()
        total_nodes = B * N

        edge_index = []
        batch = []
        for i in range(B):
            nodes = torch.arange(i * N, (i + 1) * N, device=combined_coords.device)
            edges = torch.combinations(nodes, r=2).t()
            edge_index.append(edges)
            batch.extend([i] * N)

        edge_index = torch.cat(edge_index, dim=1)
        batch = torch.tensor(batch, device=combined_coords.device)
        return edge_index, batch

"""
New Diffusion model that learns the inverse probability matrix given antigen data
"""
class DiffusionModel(nn.Module):
    def __init__(self, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.embedding_t = nn.Embedding(1000, hidden_dim)
        self.input_projection = nn.Linear(3, hidden_dim)
        self.gnn1 = GATConv(hidden_dim, hidden_dim)
        self.gnn2 = GATConv(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 3)

    def forward(self, X_t, t, edge_index, batch):
        """
        Process 3D graph data with GNN layers.

        Args:
            X_t: Noisy node features [total_nodes, 3].
            t: Time embeddings [batch_size].
            edge_index: Graph edges [2, total_edges].
            batch: Batch assignments [total_nodes].

        Returns:
            Denoised node features [total_nodes, 3].
        """
        t_emb = self.embedding_t(t)  # [batch_size, hidden_dim]
        t_emb_nodes = t_emb[batch]  # [total_nodes, hidden_dim]
        # Project input features and add time embeddings
        x = self.input_projection(X_t)  # [total_nodes, hidden_dim]
        x = x + t_emb_nodes

        # GNN layers
        x = F.relu(self.gnn1(x, edge_index))  # [total_nodes, hidden_dim]
        x = self.gnn2(x, edge_index)  # [total_nodes, hidden_dim]
        
        # Project to output space (3D coordinates)
        x = self.output_projection(x)  # [total_nodes, 3]
        return x

