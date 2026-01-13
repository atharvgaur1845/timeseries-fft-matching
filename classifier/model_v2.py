# tcn_multiclass_fault_holdout_with_confmat.py
# Usage: python tcn_multiclass_fault_holdout_with_confmat.py

import os
import itertools
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from sklearn.model_selection import StratifiedShuffleSplit

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "/home/atharv/Desktop/projects/timeseries-fft-matching/CWRU_data"
SEQ_LEN = 5000
STRIDE = 5000
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3

TRAIN_FRACS = [i / 10 for i in range(1, 10)]
REPEATS = 10

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# labels & expected keys
# -----------------------
FAULT_LABELS = {"N": 0, "IR": 1, "BA": 2, "OR": 3}
CLASS_NAMES = ["Normal", "IR", "BA", "OR"]
NUM_CLASSES = len(CLASS_NAMES)

# Known file stems we expect (but some loads may be named fault+28)
POSSIBLE_KEYS = [
    "N",
    "7IR","7BA","7OR",
    "14IR","14BA","14OR",
    "21IR","21BA","21OR",
    "IR28","BA28"   # OR28 doesn't exist in your dataset
]

FILEPATHS = {}

# -----------------------
# helpers
# -----------------------
def make_key(load, fault):
    # handle CWRU naming: 28 files are 'IR28'/'BA28' not '28IR'
    if load == 28:
        return f"{fault}28"
    else:
        return f"{load}{fault}"

def discover_files(data_dir=DATA_DIR):
    FILEPATHS.clear()
    p = Path(data_dir)
    for f in p.glob("*.csv"):
        FILEPATHS[f.stem] = str(f)
    # don't error here; we'll handle missing per-permutation
    print(f"Discovered {len(FILEPATHS)} csv files in {data_dir}")

def fault_type_from_key(key):
    if key == "N":
        return FAULT_LABELS["N"]
    if "IR" in key:
        return FAULT_LABELS["IR"]
    if "BA" in key:
        return FAULT_LABELS["BA"]
    if "OR" in key:
        return FAULT_LABELS["OR"]
    raise ValueError(f"Unknown key for label: {key}")

# -----------------------
# Dataset
# -----------------------
class SingleCSVWindowDataset(Dataset):
    def __init__(self, file_path, key):
        arr = pd.read_csv(file_path, header=None).values.squeeze().astype(np.float32)
        if arr.ndim != 1:
            arr = arr[:, 0]
        # per-file normalize to [-1,1]
        arr = 2 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) - 1
        self.data = torch.tensor(arr, dtype=torch.float32)
        self.label = fault_type_from_key(key)
        self.num_windows = max(0, (len(self.data) - SEQ_LEN) // STRIDE + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * STRIDE
        x = self.data[start:start + SEQ_LEN]
        return x.unsqueeze(0), self.label

# -----------------------
# Model (BigTCN)
# -----------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = y[:, :, :x.size(2)]
        r = x if self.down is None else self.down(x)
        return F.relu(y + r)

class BigTCN(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES):
        super().__init__()
        channels = [64, 64, 128, 128, 256]
        in_ch = 1
        layers = []
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel=5, dilation=2**i))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], n_classes)
    def forward(self, x):
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

# -----------------------
# training / evaluation
# -----------------------
def train_one_epoch(model, loader, opt):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

def eval_confusion(model, loader, n_classes=NUM_CLASSES):
    model.eval()
    conf = np.zeros((n_classes, n_classes), dtype=int)  # rows=true, cols=pred
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            y_true = yb.numpy()
            for t, p in zip(y_true, preds):
                conf[int(t), int(p)] += 1
    return conf

def per_class_accuracy_from_conf(conf):
    # conf shape (n_true, n_pred)
    true_counts = conf.sum(axis=1)
    correct = np.diag(conf)
    return correct / np.maximum(true_counts, 1)

# -----------------------
# dataset builder (handles missing files)
# -----------------------
def build_datasets(train_loads, test_load):
    train_list = []
    test_list = []

    # Normal included in both if present
    if "N" in FILEPATHS:
        train_list.append(SingleCSVWindowDataset(FILEPATHS["N"], "N"))
        test_list.append(SingleCSVWindowDataset(FILEPATHS["N"], "N"))
    else:
        print("Warning: N.csv not found.")

    # add train fault files
    for load in train_loads:
        for ft in ["IR", "BA", "OR"]:
            key = make_key(load, ft)
            if key in FILEPATHS:
                train_list.append(SingleCSVWindowDataset(FILEPATHS[key], key))
            else:
                print(f"Warning: train file missing for key {key} (skipping)")

    # add test fault files
    for ft in ["IR", "BA", "OR"]:
        key = make_key(test_load, ft)
        if key in FILEPATHS:
            test_list.append(SingleCSVWindowDataset(FILEPATHS[key], key))
        else:
            print(f"Warning: test file missing for key {key} (skipping)")

    if len(train_list) == 0 or len(test_list) == 0:
        raise RuntimeError("Not enough files found for this permutation to build datasets.")

    train_concat = ConcatDataset(train_list)
    test_concat = ConcatDataset(test_list)
    return train_concat, test_concat

# -----------------------
# confusion matrix plot
# -----------------------
def plot_confusion(conf, out_png, class_names=CLASS_NAMES):
    # conf: (n_classes, n_classes)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(conf, interpolation='nearest', cmap='Blues')
    ax.set_title("Confusion matrix (rows=true, cols=pred)")
    fig.colorbar(im, ax=ax)
    n = conf.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    # annotate counts
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved confusion matrix -> {out_png}")

# -----------------------
# single permutation run
# -----------------------
def run_experiment(train_loads, test_load):
    tag = f"train{train_loads}_test{test_load}"
    print(f"\n=== {tag} ===")

    train_full, test_full = build_datasets(train_loads, test_load)

    # labels in full train (for stratify)
    y_full = np.array([train_full[i][1] for i in range(len(train_full))])
    print("Train label distribution:", Counter(y_full))
    test_labels = [test_full[i][1] for i in range(len(test_full))]
    print("Test label distribution :", Counter(test_labels))

    test_loader = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False)

    # storage for average per-frac accuracies
    results = {c: [] for c in range(NUM_CLASSES)}

    for frac in TRAIN_FRACS:
        per_repeat_acc = {c: [] for c in range(NUM_CLASSES)}
        print(f"\n-- train frac {int(frac*100)}% --")
        for r in range(REPEATS):
            sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=SEED + r)
            try:
                train_idx, _ = next(sss.split(np.zeros(len(y_full)), y_full))
            except ValueError as e:
                print(f"StratifiedShuffleSplit failed at frac {frac} (repeat {r}): {e}")
                # fallback: random sample without stratify
                n_train = max(1, int(frac * len(y_full)))
                train_idx = np.random.RandomState(SEED + r).choice(len(y_full), n_train, replace=False)

            train_loader = DataLoader(Subset(train_full, train_idx), batch_size=BATCH_SIZE, shuffle=True)

            model = BigTCN().to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=LR)

            for ep in range(EPOCHS):
                train_one_epoch(model, train_loader, opt)

            conf = eval_confusion(model, test_loader, n_classes=NUM_CLASSES)
            accs = per_class_accuracy_from_conf(conf)
            for c in range(NUM_CLASSES):
                per_repeat_acc[c].append(accs[c])

            print(f" repeat {r+1}/{REPEATS}  per-class acc: " +
                  ", ".join([f"{CLASS_NAMES[c]}:{accs[c]:.3f}" for c in range(NUM_CLASSES)]))

        for c in range(NUM_CLASSES):
            arr = np.array(per_repeat_acc[c], dtype=float)
            results[c].append(np.nanmean(arr))
        print(" -> frac summary:", ", ".join([f"{CLASS_NAMES[c]}:{results[c][-1]:.3f}" for c in range(NUM_CLASSES)]))

    # plot per-class curves
    x = [int(f*100) for f in TRAIN_FRACS]
    plt.figure(figsize=(9,6))
    for c in range(NUM_CLASSES):
        plt.plot(x, results[c], marker='o', label=CLASS_NAMES[c])
    plt.xlabel("Training data percentage (windows from train files)")
    plt.ylabel("Per-class accuracy (test held-out load)")
    plt.title(tag)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_curve = f"{tag}_perclass_curves.png"
    plt.savefig(out_curve, dpi=200)
    print(f"Saved curve -> {out_curve}")
    plt.close()

    # Train final model on full train set and compute confusion matrix on test set
    print("Training final model on full train set to compute confusion matrix...")
    final_loader = DataLoader(train_full, batch_size=BATCH_SIZE, shuffle=True)
    final_model = BigTCN().to(DEVICE)
    final_opt = torch.optim.Adam(final_model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        train_one_epoch(final_model, final_loader, final_opt)

    final_conf = eval_confusion(final_model, test_loader, n_classes=NUM_CLASSES)
    print("Final confusion matrix (rows=true, cols=pred):\n", final_conf)
    # save confusion matrix PNG
    conf_png = f"{tag}_confusion.png"
    plot_confusion(final_conf, conf_png, class_names=CLASS_NAMES)

# -----------------------
# main
# -----------------------
def main():
    discover_files(DATA_DIR)
    loads = [7, 14, 21, 28]
    for train_pair in itertools.combinations(loads, 2):
        for test_load in [l for l in loads if l not in train_pair]:
            try:
                run_experiment(list(train_pair), test_load)
            except Exception as e:
                print(f"Skipping permutation train={train_pair} test={test_load} -> {e}")

if __name__ == "__main__":
    main()
