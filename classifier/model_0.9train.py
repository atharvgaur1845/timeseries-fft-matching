# tcn_real_vs_generated_total_mix_1824_12class.py
# Total-based generated-data replacement experiment, 12 classes

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

# ----------------------------
# CONFIG
# ----------------------------
REAL_DIR = "/home/atharv/Desktop/projects/timeseries-fft-matching/CWRU_data"
GEN_DIR  = "/home/atharv/Desktop/projects/timeseries-fft-matching/WGAN-GP/self attention"

SEQ_LEN = 1824
STRIDE  = 1824

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

TRAIN_FRAC = 0.90                      # fixed 90% regime (change if needed)
GEN_FRACS_TOTAL = [i/10 for i in range(0, 10)]  # 0.0 .. 0.9 (generated % of TOTAL train set)
REPEATS = 10

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# 12-class label mapping (files -> class id)
# keep mapping consistent with your filenames
# ----------------------------
LABEL_MAPPING = {
    "N": 0,
    "7IR": 1, "7BA": 2, "7OR": 3,
    "14IR": 4, "14BA": 5, "14OR": 6,
    "21IR": 7, "21BA": 8, "21OR": 9,
    "IR28": 10, "BA28": 11
}
CLASS_NAMES = [k for k,_ in sorted(LABEL_MAPPING.items(), key=lambda x: x[1])]  # order by id
NUM_CLASSES = len(LABEL_MAPPING)

def label_from_key(key):
    if key not in LABEL_MAPPING:
        raise KeyError(f"Unknown key/file: {key}")
    return LABEL_MAPPING[key]

# ----------------------------
# DATASETS
# ----------------------------
class RealCSVDataset(Dataset):
    """Windowed real CSV -> many windows of length SEQ_LEN (no overlap)"""
    def __init__(self, file_path, key):
        arr = pd.read_csv(file_path, header=None).values.squeeze().astype(np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        self.data = torch.tensor(arr)
        self.label = label_from_key(key)
        self.n_windows = max(0, (len(self.data) - SEQ_LEN) // STRIDE + 1)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * STRIDE
        x = self.data[start:start + SEQ_LEN]
        return x.unsqueeze(0), self.label

class GeneratedCSVDataset(Dataset):
    """
    Generated CSV: shape (1824, Ncols). Each column is an independent synthetic sample.
    Returns (1, SEQ_LEN), label
    """
    def __init__(self, file_path, key):
        arr = pd.read_csv(file_path, header=None).values.astype(np.float32)
        if arr.shape[0] != SEQ_LEN:
            raise ValueError(f"Generated file {file_path} has {arr.shape[0]} rows, expected {SEQ_LEN}")
        # per-column normalization
        arr = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-8)
        # store as torch tensor with shape (n_cols, SEQ_LEN)
        self.samples = torch.tensor(arr.T, dtype=torch.float32)
        self.label = label_from_key(key)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx].unsqueeze(0), self.label

# ----------------------------
# Big TCN (same as before)
# ----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
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
        layers = []
        in_ch = 1
        for i, ch in enumerate(chans):
            layers.append(TemporalBlock(in_ch, ch, kernel=5, dilation=2**i))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(chans[-1], NUM_CLASSES)

    def forward(self, x):
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

# ----------------------------
# training / evaluation helpers
# ----------------------------
def train_one_epoch(model, loader, opt):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

def eval_per_class_and_overall(model, loader):
    model.eval()
    correct = np.zeros(NUM_CLASSES, dtype=int)
    total = np.zeros(NUM_CLASSES, dtype=int)
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            y_true = yb.numpy()
            for p,t in zip(preds, y_true):
                total[t] += 1
                if p == t:
                    correct[t] += 1
    per_class_acc = correct / np.maximum(total, 1)
    overall = correct.sum() / max(1, total.sum())
    return per_class_acc, overall

# ----------------------------
# helpers: load directories into ConcatDataset (only include known files)
# ----------------------------
def build_real_dataset(real_dir):
    parts = []
    for f in sorted(Path(real_dir).glob("*.csv")):
        key = f.stem
        if key not in LABEL_MAPPING:
            print(f"Skipping unknown real file: {f.name}")
            continue
        ds = RealCSVDataset(str(f), key)
        if len(ds) == 0:
            print(f"Warning: real file {f.name} produced 0 windows (skip).")
            continue
        parts.append(ds)
        print(f"Real loaded: {f.name} -> class={LABEL_MAPPING[key]} windows={len(ds)}")
    if not parts:
        raise RuntimeError("No valid real CSVs found.")
    return ConcatDataset(parts)

def build_gen_dataset(gen_dir):
    parts = []
    for f in sorted(Path(gen_dir).glob("*.csv")):
        key = f.stem
        if key not in LABEL_MAPPING:
            print(f"Skipping unknown gen file: {f.name}")
            continue
        ds = GeneratedCSVDataset(str(f), key)
        if len(ds) == 0:
            print(f"Warning: gen file {f.name} produced 0 samples (skip).")
            continue
        parts.append(ds)
        print(f"Gen loaded:  {f.name} -> class={LABEL_MAPPING[key]} samples={len(ds)}")
    if not parts:
        raise RuntimeError("No valid generated CSVs found.")
    return ConcatDataset(parts)

# ----------------------------
# safe sampling utility (allow replacement when needed)
# ----------------------------
def safe_choice_from_array(arr, k, rng):
    """
    arr: 1D numpy array of indices
    k: desired number of picks
    rng: numpy RandomState for reproducibility
    returns list of selected indices (from arr values)
    """
    n_available = len(arr)
    if k <= 0:
        return []
    if n_available == 0:
        return []
    if k <= n_available:
        return rng.choice(arr, k, replace=False).tolist()
    else:
        # need replacement
        return rng.choice(arr, k, replace=True).tolist()

def safe_choice_int(n_pool, k, rng):
    """
    choose k indices from [0..n_pool-1], allow replacement if k>n_pool
    """
    if k <= 0:
        return []
    if n_pool == 0:
        return []
    if k <= n_pool:
        return rng.choice(n_pool, k, replace=False).tolist()
    else:
        return rng.choice(n_pool, k, replace=True).tolist()

# ----------------------------
# MAIN experiment (total-based generated fraction)
# ----------------------------
def main():
    print("Device:", DEVICE)
    real_ds = build_real_dataset(REAL_DIR)
    gen_ds  = build_gen_dataset(GEN_DIR)

    # label arrays for stratified splitting over real windows
    y_real = np.array([real_ds[i][1] for i in range(len(real_ds))])

    rng_global = np.random.RandomState(SEED)

    # store aggregated results: results[gf] -> list of (per_class_array, overall) across repeats
    results = {gf: [] for gf in GEN_FRACS_TOTAL}

    for gf in GEN_FRACS_TOTAL:
        print(f"\n--- Generated % of TOTAL = {int(gf*100)}% ---")
        for r in range(REPEATS):
            seed = SEED + r
            rng = np.random.RandomState(seed)

            # stratified split on real windows: train_pool (indices into real_ds), test_idx
            sss = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=seed)
            train_pool_idx, test_idx = next(sss.split(np.zeros(len(y_real)), y_real))

            # test loader (pure real)
            test_loader = DataLoader(Subset(real_ds, test_idx), batch_size=BATCH_SIZE, shuffle=False)

            total_train = len(train_pool_idx)
            n_gen = int(round(gf * total_train))     # generated count = fraction of TOTAL train pool
            n_real_needed = total_train - n_gen

            # select real windows from train_pool_idx (these are global indices in real_ds)
            real_sel = safe_choice_from_array(train_pool_idx, n_real_needed, rng)

            # select gen samples from gen_ds (indices 0..len(gen_ds)-1)
            gen_sel = safe_choice_int(len(gen_ds), n_gen, rng)

            # if both selections empty, skip (should not happen unless total_train==0)
            if (len(real_sel) + len(gen_sel)) == 0:
                print(f"Repeat {r}: empty training set (skip).")
                continue

            parts = []
            if len(real_sel) > 0:
                parts.append(Subset(real_ds, real_sel))
            if len(gen_sel) > 0:
                parts.append(Subset(gen_ds, gen_sel))

            train_concat = ConcatDataset(parts)
            train_loader = DataLoader(train_concat, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            # train model from scratch
            model = BigTCN().to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=LR)

            for ep in range(EPOCHS):
                train_one_epoch(model, train_loader, opt)

            per_class_acc, overall = eval_per_class_and_overall(model, test_loader)
            results[gf].append((per_class_acc, overall))

            print(f"gf={int(gf*100)}% rep={r+1}/{REPEATS}  overall={overall:.4f}")

    # aggregate and plot
    x = [int(g*100) for g in GEN_FRACS_TOTAL]
    plt.figure(figsize=(11,7))
    for cid in range(NUM_CLASSES):
        yvals = []
        for gf in GEN_FRACS_TOTAL:
            arrs = [r[0][cid] for r in results[gf] if r is not None]
            yvals.append(np.mean(arrs) if arrs else np.nan)
        plt.plot(x, yvals, marker='o', label=CLASS_NAMES[cid])
    # total
    total_vals = []
    for gf in GEN_FRACS_TOTAL:
        arrs = [r[1] for r in results[gf] if r is not None]
        total_vals.append(np.mean(arrs) if arrs else np.nan)
    plt.plot(x, total_vals, marker='s', linewidth=3, color='black', label='Total')

    plt.xlabel("Generated data (% of TOTAL training data)")
    plt.ylabel("Accuracy on REAL test set")
    plt.title(f"Total-based generated replacement — Train frac {int(TRAIN_FRAC*100)}%")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out = f"total_based_gen_mix_train{int(TRAIN_FRAC*100)}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved plot ->", out)

    # Save CSV summary
    rows = []
    for gf in GEN_FRACS_TOTAL:
        per_class_means = []
        per_class_stds = []
        if results[gf]:
            per_class_arr = np.stack([r[0] for r in results[gf]], axis=0)  # (repeats, classes)
            per_class_means = np.nanmean(per_class_arr, axis=0)
            per_class_stds  = np.nanstd(per_class_arr, axis=0)
            total_mean = np.nanmean([r[1] for r in results[gf]])
            total_std  = np.nanstd([r[1] for r in results[gf]])
        else:
            per_class_means = [np.nan]*NUM_CLASSES
            per_class_stds  = [np.nan]*NUM_CLASSES
            total_mean = np.nan; total_std = np.nan
        row = {"gen_frac_total": gf, "gen_percent_total": int(gf*100), "total_mean": total_mean, "total_std": total_std}
        for cid in range(NUM_CLASSES):
            row[f"class_{CLASS_NAMES[cid]}_mean"] = per_class_means[cid]
            row[f"class_{CLASS_NAMES[cid]}_std"]  = per_class_stds[cid]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_out = f"total_based_gen_mix_train{int(TRAIN_FRAC*100)}_summary.csv"
    df.to_csv(csv_out, index=False)
    print("Saved CSV ->", csv_out)
    print("DONE.")

if __name__ == "__main__":
    main()
