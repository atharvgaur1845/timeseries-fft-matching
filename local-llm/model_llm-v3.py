import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_amplitude_data_from_csv(csv_file_path, window_size=24):
    df = pd.read_csv(csv_file_path, header=None)
    raw_sensor_data = []
    for i in range(0, len(df) - window_size + 1, window_size):
        values = df.iloc[i:i+window_size, 0].tolist()
        values = [float(v) for v in values if pd.notna(v)]
        if len(values) == window_size:
            raw_sensor_data.append(values)
    return raw_sensor_data

class ContinuousSensorDataset(Dataset):
    def __init__(self, data, window_size=24):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.tensor(x, dtype=torch.float32)

class ContinuousSensorModel(nn.Module):
    def __init__(self, input_len=24, d_model=128, n_heads=4, n_layers=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, input_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)  
        x = x + self.pos_emb[:, :x.size(1)]
        for block in self.blocks:
            x = block(x)
        out = self.output_proj(x).squeeze(-1)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

def spectral_loss(pred, target):
    pred_fft = torch.fft.fft(pred, dim=-1)
    target_fft = torch.fft.fft(target, dim=-1)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

def train_continuous_model(model, dataloader, epochs=30, lr=1e-3, spectral_weight=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss1 = F.mse_loss(output, batch)
            loss2 = spectral_loss(output, batch)
            loss = (1 - spectral_weight) * loss1 + spectral_weight * loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.7f}")
    torch.save(model.state_dict(), "local-llm/best_model.pth")

def generate_bulk_synthetic_signal(model, seeds, target_total=4800):
    device = next(model.parameters()).device
    model.eval()
    all_generated = []
    with torch.no_grad():
        input_seqs = torch.tensor(seeds, dtype=torch.float32).to(device)
        predictions = model(input_seqs)
        outputs = predictions.cpu().numpy()
        all_generated.extend(outputs.reshape(-1))
    return all_generated[:target_total]
def save_data_to_csv(data, filename):
    with open(filename, "w") as f:
        for v in data:
            f.write(f"{v}\n")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == "__main__":
    csv_path = "original_data.csv"
    raw = load_amplitude_data_from_csv(csv_path, window_size=24)
    dataset = ContinuousSensorDataset(raw)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ContinuousSensorModel(input_len=24, d_model=128, n_heads=4, n_layers=8, d_ff=256, dropout=0.1)
    print(f"Model Parameters: {count_parameters(model):,}")
    train_continuous_model(model, dataloader, epochs=30, lr=1e-3)
    required_seeds = raw[:200]
    flat_generated = generate_bulk_synthetic_signal(model, required_seeds, target_total=4800)
    save_data_to_csv(flat_generated, filename="local-llm/local-llm-data-v3.csv")
