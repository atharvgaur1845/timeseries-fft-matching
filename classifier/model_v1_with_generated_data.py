# tcn_real_vs_generated_mix_1824.py
# Usage: python tcn_real_vs_generated_mix_1824.py

import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from sklearn.model_selection import StratifiedShuffleSplit

# =========================================================
# CONFIG
# =========================================================
REAL_DIR = "/home/atharv/Desktop/projects/timeseries-fft-matching/CWRU_data"
GEN_DIR  = "/home/atharv/Desktop/projects/timeseries-fft-matching/WGAN-GP/self attention"

SEQ_LEN = 1824
STRIDE  = 1824

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

TRAIN_FRACS = [i / 10 for i in range(1, 10)]      # 10–90%
GEN_FRACS   = [i / 10 for i in range(0, 11)]     # 0–100%
REPEATS = 10

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# LABELS (fault-only classification)
# =========================================================
FAULT_LABELS = {
    "N": 0,
    "IR": 1,
    "BA": 2,
    "OR": 3
}
CLASS_NAMES = ["Normal", "IR", "BA", "OR"]
NUM_CLASSES = 4

def fault_from_key(key):
    if key == "N":
        return 0
    if "IR" in key:
        return 1
    if "BA" in key:
        return 2
    if "OR" in key:
        return 3
    raise ValueError(key)

# =========================================================
# DATASETS
# =========================================================
class RealCSVDataset(Dataset):
    """Long real signal → non-overlapping windows"""
    def __init__(self, file_path, key):
        x = pd.read_csv(file_path, header=None).values.squeeze().astype(np.float32)
        if x.ndim > 1:
            x = x[:, 0]
        x = (x - x.mean()) / (x.std() + 1e-8)

        self.data = torch.tensor(x)
        self.label = fault_from_key(key)
        self.n = (len(self.data) - SEQ_LEN) // STRIDE + 1

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        s = idx * STRIDE
        return self.data[s:s+SEQ_LEN].unsqueeze(0), self.label


class GeneratedCSVDataset(Dataset):
    """1824×Ncols → each column = one sample"""
    def __init__(self, file_path, key):
        arr = pd.read_csv(file_path, header=None).values.astype(np.float32)
        assert arr.shape[0] == SEQ_LEN, "Generated data must be 1824 rows"

        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8)
        self.samples = torch.tensor(arr.T)   # (Ncols, 1824)
        self.label = fault_from_key(key)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx].unsqueeze(0), self.label

# =========================================================
# MODEL — Big TCN
# =========================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.c1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.c2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = F.relu(self.c1(x))
        y = F.relu(self.c2(y))
        y = y[:, :, :x.size(2)]
        r = x if self.down is None else self.down(x)
        return F.relu(y + r)

class BigTCN(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [64, 64, 128, 128, 256]
        layers, c = [], 1
        for i, ch in enumerate(chans):
            layers.append(TemporalBlock(c, ch, 5, 2**i))
            c = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(chans[-1], NUM_CLASSES)

    def forward(self, x):
        return self.fc(self.pool(self.tcn(x)).squeeze(-1))

# =========================================================
# TRAIN / EVAL
# =========================================================
def train_epoch(model, loader, opt):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

def eval_model(model, loader):
    model.eval()
    correct = np.zeros(NUM_CLASSES)
    total   = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            y = y.numpy()
            for pi, yi in zip(p, y):
                total[yi] += 1
                correct[yi] += (pi == yi)
    return correct / np.maximum(total, 1), correct.sum() / total.sum()

# =========================================================
# DATA LOADING
# =========================================================
def load_real():
    ds = []
    for f in Path(REAL_DIR).glob("*.csv"):
        ds.append(RealCSVDataset(f, f.stem))
    return ConcatDataset(ds)

def load_generated():
    ds = []
    for f in Path(GEN_DIR).glob("*.csv"):
        ds.append(GeneratedCSVDataset(f, f.stem))
    return ConcatDataset(ds)

# =========================================================
# MAIN EXPERIMENT
# =========================================================
def main():
    real_ds = load_real()
    gen_ds  = load_generated()

    y_real = np.array([real_ds[i][1] for i in range(len(real_ds))])
    y_gen  = np.array([gen_ds[i][1]  for i in range(len(gen_ds))])

    for tf in TRAIN_FRACS:
        results = defaultdict(list)

        for gf in GEN_FRACS:
            for r in range(REPEATS):
                sss = StratifiedShuffleSplit(1, train_size=tf, random_state=SEED+r)
                train_idx, test_idx = next(sss.split(np.zeros(len(y_real)), y_real))

                test_loader = DataLoader(Subset(real_ds, test_idx), BATCH_SIZE)

                n_train = len(train_idx)
                n_gen   = int(gf * n_train)
                n_real  = n_train - n_gen

                real_sel = np.random.choice(train_idx, n_real, replace=False) if n_real > 0 else []
                gen_sel  = np.random.choice(len(gen_ds), n_gen, replace=False) if n_gen > 0 else []

                train_parts = []
                if n_real > 0: train_parts.append(Subset(real_ds, real_sel))
                if n_gen  > 0: train_parts.append(Subset(gen_ds, gen_sel))

                train_loader = DataLoader(ConcatDataset(train_parts), BATCH_SIZE, shuffle=True)

                model = BigTCN().to(DEVICE)
                opt = torch.optim.Adam(model.parameters(), lr=LR)

                for _ in range(EPOCHS):
                    train_epoch(model, train_loader, opt)

                pc, tot = eval_model(model, test_loader)
                results[gf].append((pc, tot))

        # plot
        x = [int(g*100) for g in GEN_FRACS]
        plt.figure(figsize=(10,6))
        for c in range(NUM_CLASSES):
            plt.plot(
                x,
                [np.mean([r[0][c] for r in results[g]]) for g in GEN_FRACS],
                label=CLASS_NAMES[c]
            )
        plt.plot(x, [np.mean([r[1] for r in results[g]]) for g in GEN_FRACS],
                 color="black", linewidth=3, label="Total")
        plt.title(f"Train fraction {int(tf*100)}%")
        plt.xlabel("Generated data % in training")
        plt.ylabel("Accuracy (real test)")
        plt.grid()
        plt.legend()
        plt.savefig(f"train{int(tf*100)}_gen_mix.png", dpi=200)
        plt.close()

    print("DONE.")

if __name__ == "__main__":
    main()
