import os
import random
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
DATA_DIR = "/home/atharv/Desktop/projects/timeseries-fft-matching/CWRU_data"

SEQ_LEN = 5000          # >= 5000 values
STRIDE = 5000           # NO overlap
BATCH_SIZE = 32         # keep smaller (sequence is large)
EPOCHS = 30
LR = 1e-3

TRAIN_FRACS = [i / 10 for i in range(1, 10)]  # 10% → 90%
REPEATS = 10

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

LABEL_MAPPING = {
    "N": 0,
    "7BA": 1, "7IR": 2, "7OR": 3,
    "14BA": 4, "14IR": 5, "14OR": 6,
    "21BA": 7, "21IR": 8, "21OR": 9,
    "BA28": 10, "IR28": 11
}

NUM_CLASSES = len(LABEL_MAPPING)
CLASS_NAMES = list(LABEL_MAPPING.keys())

# =========================================================
# DATASET
# =========================================================
class SingleCSVWindowDataset(Dataset):
    """
    One CSV → many non-overlapping windows
    """
    def __init__(self, file_path, label, seq_len=5000, stride=5000):
        values = pd.read_csv(file_path, header=None).values.squeeze().astype(np.float32)

        # per-file normalization [-1, 1]
        vmin, vmax = values.min(), values.max()
        values = 2 * (values - vmin) / (vmax - vmin + 1e-8) - 1

        self.data = torch.tensor(values, dtype=torch.float32)
        self.seq_len = seq_len
        self.stride = stride
        self.label = label

        self.num_windows = (len(self.data) - seq_len) // stride

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start:start + self.seq_len]
        return x.unsqueeze(0), self.label


def build_full_dataset(csv_dir):
    datasets = []
    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue
        key = os.path.splitext(fname)[0]
        label = LABEL_MAPPING[key]
        ds = SingleCSVWindowDataset(
            os.path.join(csv_dir, fname),
            label,
            seq_len=SEQ_LEN,
            stride=STRIDE
        )
        datasets.append(ds)
        print(f"{fname:8s} | class={label:2d} | windows={len(ds)}")

    return ConcatDataset(datasets)

# =========================================================
# BIG TCN (RECEPTIVE FIELD > 5000)
# =========================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class BigTCN(nn.Module):
    """
    Receptive field ≈ kernel * sum(dilations)
    With kernel=5 and dilations up to 2^9 → > 5000
    """
    def __init__(self, n_classes):
        super().__init__()
        channels = [64, 64, 128, 128, 256, 256]
        layers = []

        in_ch = 1
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=5, dilation=dilation))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

# =========================================================
# TRAIN / EVAL
# =========================================================
def train_epoch(model, loader, opt):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate_per_class(model, loader):
    model.eval()
    correct = np.zeros(NUM_CLASSES)
    total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1).cpu().numpy()
            y = y.numpy()
            for p, t in zip(preds, y):
                total[t] += 1
                if p == t:
                    correct[t] += 1

    return correct / np.maximum(total, 1)

# =========================================================
# MAIN EXPERIMENT LOOP
# =========================================================
def main():
    full_dataset = build_full_dataset(DATA_DIR)

    labels = [full_dataset[i][1] for i in range(len(full_dataset))]
    labels = np.array(labels)

    # Store results: {train_frac: {class: [acc1, acc2, ..., acc10]}}
    results = {}

    for frac in TRAIN_FRACS:
        print(f"\n=== TRAIN FRACTION {int(frac * 100)}% ===")

        per_class_accs = {c: [] for c in range(NUM_CLASSES)}

        for r in range(REPEATS):
            # Random split for each repeat
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=frac, random_state=SEED + r
            )
            train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))

            train_ds = Subset(full_dataset, train_idx)
            test_ds = Subset(full_dataset, test_idx)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            model = BigTCN(NUM_CLASSES).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=LR)

            for ep in range(EPOCHS):
                train_epoch(model, train_loader, opt)

            acc = evaluate_per_class(model, test_loader)

            for c in range(NUM_CLASSES):
                per_class_accs[c].append(acc[c])

        results[frac] = per_class_accs
        
        # Print per-repeat summary
        for c in range(NUM_CLASSES):
            avg_acc = np.mean(per_class_accs[c])
            print(f"{CLASS_NAMES[c]:6s}: {avg_acc:.4f}")
        
        # Total accuracy
        total_avg = sum(np.mean(per_class_accs[c]) for c in range(NUM_CLASSES)) / NUM_CLASSES
        print(f"Total Accuracy: {total_avg:.4f}")
            

    # =====================================================
    # RESULTS TABLE
    # =====================================================
    print("\n" + "="*100)
    print("RESULTS TABLE - Average Accuracy across 10 runs")
    print("="*100)
    
    # Build table data
    table_data = {}
    table_data["Class"] = CLASS_NAMES + ["Total"]
    
    for frac in TRAIN_FRACS:
        frac_percent = f"{int(frac * 100)}%"
        table_data[frac_percent] = []
        
        # Per-class averages
        for c in range(NUM_CLASSES):
            avg_acc = np.mean(results[frac][c])
            table_data[frac_percent].append(f"{avg_acc:.4f}")
        
        # Total average
        total_avg = sum(np.mean(results[frac][c]) for c in range(NUM_CLASSES)) / NUM_CLASSES
        table_data[frac_percent].append(f"{total_avg:.4f}")
    
    # Create and print DataFrame
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    print("="*100)

    # =====================================================
    # PLOT
    # =====================================================
    plt.figure(figsize=(10, 6))
    x = [int(f * 100) for f in TRAIN_FRACS]

    for c in range(NUM_CLASSES):
        class_accs = [np.mean(results[frac][c]) for frac in TRAIN_FRACS]
        plt.plot(x, class_accs, marker="o", linestyle=":", label=CLASS_NAMES[c])

    # Calculate and plot total accuracy
    total_accuracies = [sum(np.mean(results[frac][c]) for c in range(NUM_CLASSES)) / NUM_CLASSES 
                        for frac in TRAIN_FRACS]
    plt.plot(x, total_accuracies, marker="s", linewidth=3, color="red", 
             label="Total Accuracy", linestyle="-")

    plt.xlabel("Training data percentage")
    plt.ylabel("Per-class accuracy")
    plt.title("Per-class Accuracy vs Training Percentage (Big TCN, seq=5000)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("tcn_5000_per_class_scaling.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
