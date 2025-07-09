import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import os
from scipy.stats import skew, kurtosis, entropy
from scipy import signal as scipy_signal
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
def compute_advanced_statistical_features(signal):
    return {
        'median': np.median(signal),
        'mad': np.median(np.abs(signal - np.median(signal))),
        'iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
        'energy': np.sum(signal**2),
        'zero_crossing_rate': len(np.where(np.diff(np.signbit(signal)))[0]),
        'autocorrelation_lag1': np.corrcoef(signal[:-1], signal[1:])[0,1] if len(signal) > 1 else 0
    }
def compute_spectrogram_features(signal, fs):
    f, t, Sxx = scipy_signal.spectrogram(signal, fs)
    spectral_centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8)
    spectral_bandwidth = np.sqrt(np.sum((f[:, np.newaxis] - spectral_centroid)**2 * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8))
    return {
        'spectral_centroid_time_avg': np.mean(spectral_centroid),
        'spectral_bandwidth_time_avg': np.mean(spectral_bandwidth),
        'spectral_contrast': np.mean(np.max(Sxx, axis=0) / (np.mean(Sxx, axis=0) + 1e-8))
    }
def compute_nonlinear_features(signal):
    def _phi(m, r, data):
        N = len(data)
        x = np.array([data[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1) / (N - m + 1)
        return np.sum(np.log(C + 1e-8)) / (N - m + 1)
    def approx_entropy(data, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(data)
        return abs(_phi(m, r, data) - _phi(m + 1, r, data))
    def sample_entropy(data, m=2, r=None):
        if r is None:
            r = 0.2 * np.std(data)
        N = len(data)
        if N <= m:
            return np.nan
        xmi = np.array([data[i:i+m] for i in range(N - m)])
        xmj = np.array([data[i:i+m] for i in range(N - m)])
        B = np.sum([np.sum(np.max(np.abs(xmi[i] - xmj), axis=1) <= r) - 1 for i in range(len(xmi))])
        m += 1
        if N <= m:
            return np.nan
        xmi1 = np.array([data[i:i+m] for i in range(N - m)])
        xmj1 = np.array([data[i:i+m] for i in range(N - m)])
        A = np.sum([np.sum(np.max(np.abs(xmi1[i] - xmj1), axis=1) <= r) - 1 for i in range(len(xmi1))])
        return -np.log(A / B) if B != 0 and A != 0 else np.nan
    def dfa_alpha(data):
        N = len(data)
        if N < 10:
            return np.nan
        Y = np.cumsum(data - np.mean(data))
        n_vals = np.floor(np.logspace(np.log10(4), np.log10(N/4), num=min(10, N//8))).astype(int)
        n_vals = n_vals[n_vals > 0]
        if len(n_vals) < 2:
            return np.nan
        F_n = []
        for n in n_vals:
            segments = N // n
            if segments == 0:
                continue
            RMS = []
            for i in range(segments):
                segment = Y[i*n:(i+1)*n]
                x = np.arange(n)
                p = np.polyfit(x, segment, 1)
                fit = np.polyval(p, x)
                RMS.append(np.sqrt(np.mean((segment - fit)**2)))
            if RMS:
                F_n.append(np.sqrt(np.mean(np.array(RMS)**2)))
        if len(F_n) < 2:
            return np.nan
        coeffs = np.polyfit(np.log(n_vals[:len(F_n)]), np.log(F_n), 1)
        return coeffs[0]
    return {
        'approximate_entropy': approx_entropy(signal),
        'sample_entropy': sample_entropy(signal),
        'detrended_fluctuation_alpha': dfa_alpha(signal)
    }
def compute_sensor_specific_features(signal, fs):
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    peak_freq_idx = np.argmax(fft_mag)
    peak_frequency = abs(freqs[peak_freq_idx])
    power = fft_mag ** 2
    power_norm = power / np.sum(power)
    freq_spread = np.sqrt(np.sum((freqs - np.sum(freqs * power_norm))**2 * power_norm))
    cumsum_power = np.cumsum(power_norm)
    rolloff_95_idx = np.where(cumsum_power >= 0.95)[0]
    rolloff_95 = abs(freqs[rolloff_95_idx[0]]) if len(rolloff_95_idx) > 0 else fs/2
    fundamental_freq = peak_frequency
    harmonic_power = 0
    total_power = np.sum(power)
    for h in range(2, 6):
        harmonic_freq = h * fundamental_freq
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
        harmonic_power += power[harmonic_idx]
    harmonic_ratio = harmonic_power / (total_power + 1e-8)
    
    return {
        'peak_frequency': peak_frequency,
        'frequency_spread': freq_spread,
        'spectral_rolloff_95': rolloff_95,
        'harmonic_ratio': harmonic_ratio,
        'spectral_flux': np.sum(np.diff(fft_mag)**2),
        'spectral_decrease': np.sum((fft_mag[1:] - fft_mag[0]) / np.arange(1, len(fft_mag)))
    }
def extract_combined_features(signal, fs):
    time_feats = compute_time_domain_features(signal)
    freq_feats = compute_frequency_domain_features(signal, fs)
    advanced_feats = compute_advanced_statistical_features(signal)
    spectrogram_feats = compute_spectrogram_features(signal, fs)
    nonlinear_feats = compute_nonlinear_features(signal)
    sensor_feats = compute_sensor_specific_features(signal, fs)
    all_features = {**time_feats, **freq_feats, **advanced_feats, 
                   **spectrogram_feats, **nonlinear_feats, **sensor_feats}
    
    for key, value in all_features.items():
        if np.isnan(value) or np.isinf(value):
            all_features[key] = 0.0
    
    return all_features
def load_amplitude_data_from_csv(csv_file_path, window_size=256):
    df = pd.read_csv(csv_file_path, header=None)
    raw_sensor_data = []
    for i in range(0, len(df) - window_size + 1, window_size):
        values = df.iloc[i:i+window_size, 0].tolist()
        values = [float(v) for v in values if pd.notna(v)]
        if len(values) == window_size:
            raw_sensor_data.append(values)
    return raw_sensor_data
class ContinuousSensorDataset(Dataset):
    def __init__(self, data, window_size=256, fs=12000):
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
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
    def forward(self, x):
        B, T, D = x.size()
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x
class VariationalLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, training=True):
        if not training:
            return x
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar
class ContinuousSensorModel(nn.Module):
    def __init__(self, input_len=256, d_model=256, n_heads=8, n_layers=12, d_ff=512, dropout=0.1, feature_dim=40):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, input_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.variational = VariationalLayer(d_model)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.input_residual = nn.Linear(1, 1)

    def forward(self, x, feats, training=True):
        B, T = x.size()
        x_residual = self.input_residual(x.unsqueeze(-1)).squeeze(-1)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T, :]
        f = self.feature_proj(feats)
        feature_gate = torch.sigmoid(f.unsqueeze(1))
        x = x * (1 + 0.1 * feature_gate.expand(-1, T, -1))
        for block in self.blocks:
            x = block(x)
        if training:
            x, mu, logvar = self.variational(x, training)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (B * T)
        else:
            x = self.variational(x, training)
            kl_loss = 0
        out = self.output_proj(x).squeeze(-1)
        out = out + 0.1 * x_residual
        if training:
            return out, kl_loss
        return out
def variance_preserving_loss(pred, target):
    pred_var = torch.var(pred, dim=-1)
    target_var = torch.var(target, dim=-1)
    return F.mse_loss(pred_var, target_var)
def gradient_penalty_loss(pred, target):
    pred_grad = torch.diff(pred, dim=-1)
    target_grad = torch.diff(target, dim=-1)
    return F.mse_loss(pred_grad, target_grad)
def spectral_loss(pred, target):
    pred_fft = torch.fft.fft(pred, dim=-1)
    target_fft = torch.fft.fft(target, dim=-1)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
def correlation_loss(pred, target):
    vx = pred - pred.mean(dim=-1, keepdim=True)
    vy = target - target.mean(dim=-1, keepdim=True)
    corr = (vx * vy).sum(dim=-1) / (torch.sqrt((vx**2).sum(dim=-1)) * torch.sqrt((vy**2).sum(dim=-1)) + 1e-8)
    return 1 - corr.mean()
def train_enhanced_model(model, dataloader, feat_mean, feat_std, epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_kl = 0
        for batch in dataloader:
            seqs, feats = batch
            seqs, feats = seqs.to(device), feats.to(device)
            feats = (feats - feat_mean.to(device)) / feat_std.to(device)
            optimizer.zero_grad()
            output, kl_loss = model(seqs, feats, training=True)
            mse_loss = F.mse_loss(output, seqs)
            spec_loss = spectral_loss(output, seqs)
            var_loss = variance_preserving_loss(output, seqs)
            grad_loss = gradient_penalty_loss(output, seqs)
            total_loss_batch = (
                0.4 * mse_loss + 
                0.3 * spec_loss + 
                0.2 * var_loss + 
                0.1 * grad_loss + 
                0.01 * kl_loss
            )
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += total_loss_batch.item()
            total_kl += kl_loss if isinstance(kl_loss, (int, float)) else kl_loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, KL: {avg_kl:.6f}, LR: {current_lr:.2e}")
def generate_dynamic_synthetic_signal(model, seeds, feat_mean, feat_std, fs=12000, target_total=4800, temperature=1.0, diversity_scale=0.1):
    device = next(model.parameters()).device
    model.eval()
    all_generated = []
    with torch.no_grad():
        for i, seg in enumerate(seeds):
            feats = extract_combined_features(np.array(seg), fs)
            seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            feat_tensor = torch.tensor(list(feats.values()), dtype=torch.float32).unsqueeze(0).to(device)
            feat_tensor = (feat_tensor - feat_mean.to(device)) / feat_std.to(device)
            pred = model(seg_tensor, feat_tensor, training=False)
            if temperature != 1.0:
                pred = pred / temperature
            if diversity_scale > 0:
                noise = torch.randn_like(pred) * diversity_scale * torch.std(pred)
                pred = pred + noise
            pred_np = pred.cpu().numpy().flatten()
            all_generated.extend(pred_np)
            if len(all_generated) >= target_total:
                break
    
    return all_generated[:target_total]

def save_data_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for v in data:
            f.write(f"{v}\n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    csv_path = "original_data.csv"
    window_size = 256
    raw = load_amplitude_data_from_csv(csv_path, window_size=window_size)
    dataset = ContinuousSensorDataset(raw, window_size=window_size)
    sample_features = extract_combined_features(np.random.randn(window_size), 12000)
    feature_count = len(sample_features)
    print(f"Enhanced feature count: {feature_count} features")
    all_feats = torch.stack([feat for _, feat in dataset])
    feat_mean = all_feats.mean(dim=0)
    feat_std = all_feats.std(dim=0) + 1e-6
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ContinuousSensorModel(
        input_len=window_size,
        d_model=256, 
        n_heads=8,
        n_layers=12,
        d_ff=512,
        dropout=0.1,
        feature_dim=feature_count
    )
    
    print(f"Enhanced Model Parameters: {count_parameters(model):,}")
    train_enhanced_model(model, dataloader, feat_mean, feat_std, epochs=50, lr=1e-4)
    required_seeds = raw[:200]
    flat_generated = generate_dynamic_synthetic_signal(
        model, required_seeds, feat_mean, feat_std, 
        target_total=4800, temperature=1.1, diversity_scale=0.05
    )
    save_data_to_csv(flat_generated, filename="local-llm/enhanced-llm-data-v5.csv")
