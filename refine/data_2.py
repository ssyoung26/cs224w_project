import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random

alphabet = '#ACDEFGHIKLMNPQRSTVWY'  # Amino acid alphabet
DUMMY = {
    'pdb': None,
    'antibody_seq': '#' * 10,
    'antigen_seq': '#' * 10,
    'antibody_coords': np.zeros((10, 3)) + np.nan,
    'antigen_coords': np.zeros((10, 3)) + np.nan,
    'antibody_cdr': '#' * 10,
    'antibody_atypes': [0] * 10,
    'antigen_atypes': [0] * 10,
}

class AntibodyDataset2:
    def __init__(self, jsonl_file, cdr_type='3', max_len=130):
        self.data = []
        with open(jsonl_file) as f:
            lines = f.readlines()
            for line in lines:
                entry = json.loads(line)

                # Skip entries without antibody CDRs
                if entry['antibody_cdr'] is None or cdr_type not in entry['antibody_cdr']:
                    continue

                # Truncate antibody information based on CDR location and max_len
                last_cdr = entry['antibody_cdr'].rindex(cdr_type)
                if last_cdr >= max_len - 1:
                    entry['antibody_seq'] = entry['antibody_seq'][last_cdr - max_len + 10 : last_cdr + 10]
                    entry['antibody_cdr'] = entry['antibody_cdr'][last_cdr - max_len + 10 : last_cdr + 10]
                    entry['antibody_coords'] = entry['antibody_coords'][last_cdr - max_len + 10 : last_cdr + 10]
                    entry['antibody_atypes'] = entry['antibody_atypes'][last_cdr - max_len + 10 : last_cdr + 10]
                else:
                    entry['antibody_seq'] = entry['antibody_seq'][:max_len]
                    entry['antibody_cdr'] = entry['antibody_cdr'][:max_len]
                    entry['antibody_coords'] = entry['antibody_coords'][:max_len]
                    entry['antibody_atypes'] = entry['antibody_atypes'][:max_len]

                # Truncate antigen information
                entry['antigen_seq'] = entry['antigen_seq'][:max_len]
                entry['antigen_coords'] = entry['antigen_coords'][:max_len]
                entry['antigen_atypes'] = entry['antigen_atypes'][:max_len]

                # Append valid entry
                if len(entry['antibody_seq']) > 0 and len(entry['antigen_seq']) > 0:
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader2:
    def __init__(self, dataset, batch_tokens, interval_sort=0):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['antibody_seq']) for i in range(self.size)]
        self.batch_tokens = batch_tokens

        if interval_sort > 0:
            cdr_type = str(interval_sort)
            self.lengths = [dataset[i]['antibody_cdr'].count(cdr_type) for i in range(self.size)]
            self.intervals = [
                (dataset[i]['antibody_cdr'].index(cdr_type), dataset[i]['antibody_cdr'].rindex(cdr_type))
                for i in range(self.size)
            ]
            sorted_ix = sorted(range(self.size), key=self.intervals.__getitem__)
        else:
            sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_tokens:
                batch.append(ix)
            else:
                clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def completize_data(batch):
    B = len(batch)
    L_antibody = max(len(b['antibody_seq']) for b in batch)  # Max antibody length
    L_antigen = max(len(b['antigen_seq']) for b in batch)    # Max antigen length

    X_antibody = np.zeros([B, L_antibody, 14, 3])
    S_antibody = np.zeros([B, L_antibody], dtype=np.int32)
    mask_antibody = np.zeros([B, L_antibody], dtype=np.float32)

    X_antigen = np.zeros([B, L_antigen, 14, 3])
    S_antigen = np.zeros([B, L_antigen], dtype=np.int32)

    for i, b in enumerate(batch):
        antibody_coords = np.array(b['antibody_coords'])
        X_antibody[i, :antibody_coords.shape[0], :, :] = antibody_coords  # Assign coords
        S_antibody[i, :len(b['antibody_seq'])] = [alphabet.index(a) for a in b['antibody_seq']]
        mask_antibody[i, :len(b['antibody_seq'])] = 1.0

        antigen_coords = np.array(b['antigen_coords'])
        X_antigen[i, :antigen_coords.shape[0], :, :] = antigen_coords  # Assign coords
        S_antigen[i, :len(b['antigen_seq'])] = [alphabet.index(a) for a in b['antigen_seq']]

    mask_antibody *= np.isfinite(np.sum(X_antibody, axis=(2, 3))).astype(np.float32)
    X_antibody[np.isnan(X_antibody)] = 0.0
    X_antigen[np.isnan(X_antigen)] = 0.0

    X_antibody = torch.from_numpy(X_antibody).float().cuda()
    S_antibody = torch.from_numpy(S_antibody).long().cuda()
    mask_antibody = torch.from_numpy(mask_antibody).float().cuda()

    X_antigen = torch.from_numpy(X_antigen).float().cuda()
    S_antigen = torch.from_numpy(S_antigen).long().cuda()

    return X_antibody, S_antibody, mask_antibody, X_antigen, S_antigen

