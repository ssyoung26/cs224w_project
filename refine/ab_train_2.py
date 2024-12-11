import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os
from torch.cuda.amp import GradScaler, autocast

from structgen import *
from tqdm import tqdm


def evaluate(model, loader, args):
    model.eval()
    val_nll = val_tot = 0.0
    val_rmsd = []
    with torch.no_grad():
        for hbatch in tqdm(loader):
            X_antibody, S_antibody, mask_antibody, X_antigen, S_antigen = completize_data(hbatch)
            antibody_cdr = [b['antibody_cdr'] for b in hbatch]

            for i in range(len(hbatch)):
                L = mask_antibody[i:i+1].sum().long().item()
                if L > 0:
                    out = model.log_prob(
                        S_antibody[i:i+1, :L].long(),
                        [antibody_cdr[i]],
                        mask_antibody[i:i+1, :L],
                        antigen_coords=X_antigen[i:i+1, :],
                        antigen_seq=S_antigen[i:i+1, :]
                    )
                    nll, X_pred = out.nll, out.X_cdr
                    val_nll += nll.item() * antibody_cdr[i].count(args.cdr_type)
                    val_tot += antibody_cdr[i].count(args.cdr_type)
                    l, r = antibody_cdr[i].index(args.cdr_type), antibody_cdr[i].rindex(args.cdr_type)
                    rmsd = compute_rmsd(
                        X_pred[:, :, 1, :],  # predicted alpha carbons
                        X_antibody[i:i+1, l:r+1, 1, :],  # ground truth alpha carbons
                        mask_antibody[i:i+1, l:r+1]
                    )
                    val_rmsd.append(rmsd.item())

    val_ppl = math.exp(val_nll / val_tot) if val_tot > 0 else float('inf')
    avg_rmsd = sum(val_rmsd) / len(val_rmsd) if val_rmsd else float('inf')
    return val_ppl, avg_rmsd

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='data/sabdab_2022_01/train_data.jsonl')
parser.add_argument('--val_path', default='data/sabdab_2022_01/val_data.jsonl')
parser.add_argument('--test_path', default='data/sabdab_2022_01/test_data.jsonl')
parser.add_argument('--save_dir', default='ckpts/tmp')
parser.add_argument('--load_model', default=None)

parser.add_argument('--cdr_type', default='3')

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=100)
parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--block_size', type=int, default=8)
parser.add_argument('--update_freq', type=int, default=1)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=21)
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)

args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

loaders = []
for path in [args.train_path, args.val_path, args.test_path]:
    data = AntibodyDataset2(path, cdr_type=args.cdr_type)
    loader = StructureLoader2(data.data, batch_tokens=args.batch_tokens, interval_sort=int(args.cdr_type))
    loaders.append(loader)

loader_train, loader_val, loader_test = loaders

model = HierarchicalDecoder2(args).cuda()

optimizer = torch.optim.Adam(model.parameters())
if args.load_model:
    model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
    model = HierarchicalDecoder2(model_args).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

print('Training:{}, Validation:{}, Test:{}'.format(
    len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
)

best_ppl, best_epoch = 100, -1

scaler = GradScaler()

for e in range(args.epochs):
    model.train()
    meter = 0

    for i, hbatch in enumerate(tqdm(loader_train)):
        optimizer.zero_grad()
        X_antibody, S_antibody, mask_antibody, X_antigen, S_antigen = completize_data(hbatch)
        antibody_cdr = [b['antibody_cdr'] for b in hbatch]

        # loss, snll = model(X_antibody, S_antibody, antibody_cdr, mask_antibody, antigen_coords=X_antigen, antigen_seq=S_antigen)

        # loss.backward()
        # optimizer.step()

        with autocast():
            loss, snll = model(
                X_antibody.float(), S_antibody.long(), antibody_cdr, mask_antibody,
                antigen_coords=X_antigen.float(), antigen_seq=S_antigen.float()
            )
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        meter += snll.exp().item()
        if (i + 1) % args.print_iter == 0:
            meter /= args.print_iter
            print(f'[{i + 1}] Train PPL = {meter:.3f}')
            meter = 0

    val_ppl, val_rmsd = evaluate(model, loader_val, args)
    ckpt = (model.state_dict(), optimizer.state_dict(), args)
    torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
    print(f'Epoch {e}, Val PPL = {val_ppl:.3f}, Val RMSD = {val_rmsd:.3f}')

    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_epoch = e

if best_epoch >= 0:
    best_ckpt = os.path.join(args.save_dir, f"model.ckpt.{best_epoch}")
    model.load_state_dict(torch.load(best_ckpt)[0])

test_ppl, test_rmsd = evaluate(model, loader_test, args)
print(f'Test PPL = {test_ppl:.3f}, Test RMSD = {test_rmsd:.3f}')

