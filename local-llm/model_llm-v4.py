import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import os
from scipy.stats import skew, kurtosis, entropy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def compute_time_domain_features(signal):
    mean_val = np.mean(signal)
    std_dev = np.std(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    abs_mean = np.mean(np.abs(signal))
    peak_val = np.max(np.abs(signal))
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    var_val = np.var(signal)
    kurt_index = kurt_val / (rms**4 + 1e-8)
    peak_index = peak_val / (rms + 1e-8)
    waveform_index = rms / (abs_mean + 1e-8)
    pulse_index = peak_val / (abs_mean + 1e-8)
    return {
        'mean_value': mean_val,
        'standard_deviation': std_dev,
        'square_root_amplitude': rms,
        'absolute_mean_value': abs_mean,
        'peak_value': peak_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'variance': var_val,
        'kurtosis_index': kurt_index,
        'peak_index': peak_index,
        'waveform_index': waveform_index,
        'pulse_index': pulse_index
    }

def compute_frequency_domain_features(signal, fs):
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals)
    fft_freqs = np.fft.fftfreq(len(signal), d=1/fs)
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    spectrum = fft_mag[pos_mask]
    norm_spec = spectrum / (np.sum(spectrum) + 1e-8)
    centroid = np.sum(freqs * norm_spec)
    spread = np.sqrt(np.sum((freqs - centroid) ** 2 * norm_spec))
    rolloff = freqs[np.where(np.cumsum(norm_spec) >= 0.85)[0][0]]
    energy = np.sum(spectrum ** 2)
    peak = np.max(spectrum)
    flatness = np.exp(np.mean(np.log(spectrum + 1e-8))) / (np.mean(spectrum) + 1e-8)
    return {
        'frequency_mean_value': np.mean(spectrum),
        'frequency_variance': np.var(spectrum),
        'frequency_skewness': skew(spectrum),
        'frequency_kurtosis': kurtosis(spectrum),
        'frequency_standard_deviation': np.std(spectrum),
        'frequency_root_mean_square': np.sqrt(np.mean(spectrum ** 2)),
        'average_frequency': centroid,
        'gravity_frequency': centroid,
        'regularity_degree': spread,
        'variation_parameter': spread / (centroid + 1e-8),
        'eighth_order_moment': np.mean(spectrum ** 8),
        'sixteenth_order_moment': np.mean(spectrum ** 16),
        'entropy': entropy(norm_spec),
        'spectral_rolloff_85': rolloff,
        'spectral_energy': energy,
        'spectral_peak': peak,
        'spectral_flatness': flatness
    }

def extract_combined_features(signal, fs):
    time_feats = compute_time_domain_features(signal)
    freq_feats = compute_frequency_domain_features(signal, fs)
    return {**time_feats, **freq_feats}

def load_amplitude_data_from_csv(csv_file_path, window_size=1024):
    df = pd.read_csv(csv_file_path, header=None)
    raw_sensor_data = []
    for i in range(0, len(df) - window_size + 1, window_size):
        values = df.iloc[i:i+window_size, 0].tolist()
        values = [float(v) for v in values if pd.notna(v)]
        if len(values) == window_size:
            raw_sensor_data.append(values)
    return raw_sensor_data

class ContinuousSensorDataset(Dataset):
    def __init__(self, data, window_size=24, fs=12000):
        self.samples = []
        for segment in data:
            if len(segment) == window_size:
                feat = extract_combined_features(np.array(segment), fs)
                self.samples.append((segment, list(feat.values())))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, feats = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(feats, dtype=torch.float32)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
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

class ContinuousSensorModel(nn.Module):
    def __init__(self, input_len=128, d_model=256, n_heads=4, n_layers=8, d_ff=256, dropout=0.1, feature_dim=40):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, input_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, feats):
        x = x.unsqueeze(-1)
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1)]
        f = self.feature_proj(feats).unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + f
        for block in self.blocks:
            x = block(x)
        out = self.output_proj(x).squeeze(-1)
        return out

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
            seqs, feats = batch
            seqs, feats = seqs.to(device), feats.to(device)
            optimizer.zero_grad()
            output = model(seqs, feats)
            loss1 = F.mse_loss(output, seqs)
            loss2 = spectral_loss(output, seqs)
            loss = (1 - spectral_weight) * loss1 + spectral_weight * loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")
    torch.save(model.state_dict(), "local-llm/best_model.pth")

def generate_bulk_synthetic_signal(model, seeds, fs=12000, target_total=4800):
    device = next(model.parameters()).device
    model.eval()
    all_generated = []
    with torch.no_grad():
        for seg in seeds:
            feats = extract_combined_features(np.array(seg), fs)
            seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            feat_tensor = torch.tensor(list(feats.values()), dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seg_tensor, feat_tensor).cpu().numpy().flatten()
            all_generated.extend(pred)
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
    first_feat = extract_combined_features(np.array(raw[0]), fs=12000)
    feature_dim = len(first_feat)

    dataset = ContinuousSensorDataset(raw)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ContinuousSensorModel(
        input_len=24,
        d_model=128,
        n_heads=4,
        n_layers=8,
        d_ff=256,
        dropout=0.1,
        feature_dim=feature_dim
    )

    print(f"Model Parameters: {count_parameters(model):,}")
    train_continuous_model(model, dataloader, epochs=30, lr=1e-3)
    required_seeds = raw[:200]
    flat_generated = generate_bulk_synthetic_signal(model, required_seeds, target_total=4800)
    save_data_to_csv(flat_generated, filename="local-llm/local-llm-data-v4.csv")
