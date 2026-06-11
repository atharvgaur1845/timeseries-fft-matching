import os
import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import math

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = 'CWRU_data'  # Folder containing your CSVs
SEQ_LEN = 1824
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Training Configs
GAN_EPOCHS = 50      # Epochs to train GAN per iteration
CLS_EPOCHS = 30      # Epochs to train Classifier per iteration
LR_GAN = 1e-4
LR_CLS = 1e-3

# WGAN-GP Hyperparameters
NZ = 100             # Latent vector size
LAMBDA_GP = 10       # Gradient penalty lambda

# Experiments: 10% Real (+80% Syn) -> ... -> 90% Real (+0% Syn)
REAL_FRACTIONS = [i / 10 for i in range(1, 10)] 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================================================
# DATASET
# =========================================================
LABEL_MAPPING = {
    "N": 0, "7BA": 1, "7IR": 2, "7OR": 3, 
    "14BA": 4, "14IR": 5, "14OR": 6, 
    "21BA": 7, "21IR": 8, "21OR": 9, 
    "BA28": 10, "IR28": 11
}
NUM_CLASSES = len(LABEL_MAPPING)

class CWRUDataset(Dataset):
    def __init__(self, file_path, seq_len=1824):
        filename = os.path.splitext(os.path.basename(file_path))[0].replace("_Sensor1", "")
        label = -1
        for k, v in LABEL_MAPPING.items():
            if k == filename or (k in filename and len(k) > 1):
                label = v
                break
        if label == -1:
            if "Normal" in filename: label = 0
            else: raise ValueError(f"Could not map filename {filename} to label")

        self.label = label
        df = pd.read_csv(file_path, header=None)
        raw_values = df.values.flatten().astype(np.float32)
        
        # Normalize
        self.min_val = np.min(raw_values)
        self.max_val = np.max(raw_values)
        values = 2 * (raw_values - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1
        
        self.samples = []
        n_samples = len(values) // seq_len
        for i in range(n_samples):
            self.samples.append(values[i*seq_len : (i+1)*seq_len])
            
        self.data = torch.tensor(np.array(self.samples))
        self.labels = torch.full((len(self.data),), self.label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0), self.labels[idx]

def load_all_data(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    datasets = []
    for f in files:
        try:
            ds = CWRUDataset(f, SEQ_LEN)
            if len(ds) > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return ConcatDataset(datasets)

class SyntheticDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# =========================================================
# MODEL: GAN COMPONENTS (From model_self_attention.py)
# =========================================================

class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction  
        self.query = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, length = x.size()
        Q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        K = self.key(x).view(batch_size, -1, length)
        V = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)
        attention = torch.bmm(Q, K) / math.sqrt(self.in_channels // self.reduction)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(attention, V).permute(0, 2, 1).view(batch_size, channels, length)
        return self.gamma * out + x

class AttentionTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.attention = SelfAttention(out_channels)
    
    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + res
        x = self.attention(x)
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.dropout(x)
        return x + res

class GeneratorTCNWithSelfAttention(nn.Module):
    def __init__(self, nz=100, num_classes=12, embed_size=10, num_blocks=9, 
                 channels=64, kernel_size=5, dropout=0.05, output_length=1824):
        super().__init__()
        self.output_length = output_length
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, embed_size)
        self.initial_length = 114
        
        self.init_conv = nn.ConvTranspose1d(nz + embed_size, channels, kernel_size=self.initial_length, stride=1, padding=0)
        self.init_bn = nn.BatchNorm1d(channels)
        self.init_relu = nn.ReLU(True)

        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(channels), nn.ReLU(True),
        )

        self.tcn_blocks = nn.Sequential(*[
            AttentionTCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.global_attention = SelfAttention(channels)
        
        self.downsample_layers = nn.Sequential(
            nn.Conv1d(channels, channels//2, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(channels//2), nn.ReLU(True),
            nn.Conv1d(channels//2, channels//4, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(channels//4), nn.ReLU(True),
            nn.Conv1d(channels//4, channels//8, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(channels//8), nn.ReLU(True),
            nn.Conv1d(channels//8, 1, kernel_size=1, stride=1, padding=0),
        )
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1).unsqueeze(2)
        x = self.init_relu(self.init_bn(self.init_conv(x)))
        x = self.upsample_layers(x)
        
        if x.size(2) != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
            
        x = self.tcn_blocks(x)
        x = self.global_attention(x)
        x = self.downsample_layers(x)
        
        if x.size(2) != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
            
        return self.tanh(x)

class DiscriminatorTCN(nn.Module):
    def __init__(self, num_classes=12, num_blocks=9, channels=64, kernel_size=5, dropout=0.05):
        super().__init__()
        self.initial_conv = nn.Conv1d(1, channels, kernel_size=1)
        self.tcn_layers = nn.Sequential(*[TCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout) for i in range(num_blocks)])
        self.flatten = nn.AdaptiveAvgPool1d(1)
        self.adv_output = nn.Linear(channels, 1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.tcn_layers(x)
        x = self.flatten(x).squeeze(2)
        return self.adv_output(x), self.classifier(x)

# =========================================================
# MODEL: CLASSIFIER (From model_v1_with_generated_data.py)
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
        y = y[:, :, :x.size(2)] # Causal cropping
        r = x if self.down is None else self.down(x)
        return F.relu(y + r)

class BigTCN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        chans = [64, 64, 128, 128, 256]
        layers, c = [], 1
        for i, ch in enumerate(chans):
            layers.append(TemporalBlock(c, ch, 5, 2**i))
            c = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(chans[-1], num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.tcn(x)).squeeze(-1))

# =========================================================
# TRAINING FUNCTIONS
# =========================================================

def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1).to(device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp, _ = D(interp)
    grads = autograd.grad(outputs=d_interp, inputs=interp,
                          grad_outputs=torch.ones_like(d_interp),
                          create_graph=True, retain_graph=True)[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def train_gan(train_loader, epochs=GAN_EPOCHS):
    netG = GeneratorTCNWithSelfAttention(nz=NZ, num_classes=NUM_CLASSES).to(DEVICE)
    netD = DiscriminatorTCN(num_classes=NUM_CLASSES).to(DEVICE)
    optG = optim.Adam(netG.parameters(), lr=LR_GAN, betas=(0.5, 0.9))
    optD = optim.Adam(netD.parameters(), lr=LR_GAN, betas=(0.5, 0.9))
    
    print(f"   -> Training GAN for {epochs} epochs (Heavy Model)...")
    netG.train(); netD.train()
    
    for epoch in range(epochs):
        for real_data, labels in train_loader:
            real_data, labels = real_data.to(DEVICE), labels.to(DEVICE)
            curr_bs = real_data.size(0)

            # Train Discriminator
            optD.zero_grad()
            noise = torch.randn(curr_bs, NZ, device=DEVICE)
            fake_data = netG(noise, labels).detach()
            
            d_real, c_real = netD(real_data)
            d_fake, _      = netD(fake_data)
            
            gp = gradient_penalty(netD, real_data, fake_data, DEVICE)
            loss_d = (torch.mean(d_fake) - torch.mean(d_real)) + LAMBDA_GP * gp + F.cross_entropy(c_real, labels)
            
            loss_d.backward()
            optD.step()

            # Train Generator (Every step for simplicity in pipeline)
            optG.zero_grad()
            noise = torch.randn(curr_bs, NZ, device=DEVICE)
            gen_data = netG(noise, labels)
            d_gen, c_gen = netD(gen_data)
            
            loss_g = -torch.mean(d_gen) + F.cross_entropy(c_gen, labels)
            loss_g.backward()
            optG.step()
                
    return netG

def generate_synthetic_data(generator, num_samples):
    generator.eval()
    syn_data_list = []
    syn_label_list = []
    
    samples_per_class = num_samples // NUM_CLASSES
    remainder = num_samples % NUM_CLASSES
    
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            count = samples_per_class + (1 if c < remainder else 0)
            if count == 0: continue
            
            noise = torch.randn(count, NZ, device=DEVICE)
            labels = torch.full((count,), c, dtype=torch.long, device=DEVICE)
            
            fake = generator(noise, labels).cpu()
            syn_data_list.append(fake)
            syn_label_list.append(labels.cpu())
            
    return SyntheticDataset(torch.cat(syn_data_list), torch.cat(syn_label_list))

def train_classifier(loader, epochs=CLS_EPOCHS):
    model = BigTCN(num_classes=NUM_CLASSES).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR_CLS)
    crit = nn.CrossEntropyLoss()
    
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
    return model

def evaluate_classifier(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading Data...")
    full_dataset = load_all_data(DATA_DIR)
    
    labels = [y.item() for _, y in full_dataset]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_indices, test_indices = next(sss.split(np.zeros(len(labels)), labels))
    
    global_test_ds = Subset(full_dataset, test_indices)
    test_loader = DataLoader(global_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    total_train_slots = len(train_indices)
    print(f"Global Train Pool: {total_train_slots}, Global Test: {len(test_indices)}")
    
    results = []
    
    for real_frac in REAL_FRACTIONS:
        n_real = int(total_train_slots * real_frac) 
        n_syn = total_train_slots - n_real
        
        print(f"\n=== Experiment: {int(real_frac*100)}% Real ({n_real}) + {int((1-real_frac)*100)}% Syn ({n_syn}) ===")
        
        train_labels = [full_dataset[i][1].item() for i in train_indices]
        sss_sub = StratifiedShuffleSplit(n_splits=1, train_size=n_real, random_state=SEED)
        sub_idx, _ = next(sss_sub.split(np.zeros(len(train_indices)), train_labels))
        real_subset_indices = [train_indices[i] for i in sub_idx]
        
        real_ds = Subset(full_dataset, real_subset_indices)
        real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        if n_syn > 0:
            print(f"   [1/3] Training GAN on {n_real} samples...")
            generator = train_gan(real_loader)
            
            print(f"   [2/3] Generating {n_syn} synthetic samples...")
            syn_ds = generate_synthetic_data(generator, n_syn)
            combined_ds = ConcatDataset([real_ds, syn_ds])
        else:
            print("   [1/3] Skipping GAN (100% Real Data)")
            combined_ds = real_ds

        print(f"   [3/3] Training Classifier on {len(combined_ds)} samples...")
        combined_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True)
        classifier = train_classifier(combined_loader)
        
        acc = evaluate_classifier(classifier, test_loader)
        results.append((int(real_frac*100), acc))
        print(f"   >>> Accuracy: {acc*100:.2f}%")
        
    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linewidth=2)
    plt.title("Classifier Accuracy vs Real Data Percentage\n(Remainder filled with Syn Data)")
    plt.xlabel("% of Real Data")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.savefig("gan_augmentation_results.png")
    pd.DataFrame({'Real_Percent': x_vals, 'Accuracy': y_vals}).to_csv("experiment_results.csv", index=False)
    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()