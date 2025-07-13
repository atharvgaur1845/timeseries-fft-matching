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

def advanced_statistical_features(signal):
    return {
        'median': np.median(signal),
        'mad': np.median(np.abs(signal - np.median(signal))),
        'iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
        'energy': np.sum(signal**2),
        'zero_crossing_rate': len(np.where(np.diff(np.signbit(signal)))[0]),
        'autocorrelation_lag1': np.corrcoef(signal[:-1], signal[1:])[0,1] if len(signal) > 1 else 0
    }
def spectrogram_features(signal, fs):
    f, t, Sxx = scipy_signal.spectrogram(signal, fs)
    spectral_centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8)
    spectral_bandwidth = np.sqrt(np.sum((f[:, np.newaxis] - spectral_centroid)**2 * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8))
    
    return {
        'spectral_centroid_time_avg': np.mean(spectral_centroid),
        'spectral_bandwidth_time_avg': np.mean(spectral_bandwidth),
        'spectral_contrast': np.mean(np.max(Sxx, axis=0) / (np.mean(Sxx, axis=0) + 1e-8))
    }
def nonlinear_features(signal):
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

def sensor_specific_features(signal, fs):
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
    advanced_feats = advanced_statistical_features(signal)
    spectrogram_feats = spectrogram_features(signal, fs)
    nonlinear_feats = nonlinear_features(signal)
    sensor_feats = sensor_specific_features(signal, fs)
    
    all_features = {**time_feats, **freq_feats, **advanced_feats, 
                   **spectrogram_feats, **nonlinear_feats, **sensor_feats}
    for key, value in all_features.items():
        if np.isnan(value) or np.isinf(value):
            all_features[key] = 0.0
    
    return all_features
def load_data_from_csv(csv_file_path, window_size=256):
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

def save_data_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for v in data:
            f.write(f"{v}\n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class StatisticalMatchingHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.stats_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 5)
        )
        
    def forward(self, x):
        pooled = torch.mean(x, dim=1)
        stats = self.stats_predictor(pooled)
        return stats

class MultiHead_Model(nn.Module):

    def __init__(self, input_len=256, d_model=512, n_heads=8, n_layers=8, d_ff=1024, dropout=0.05, feature_dim=47):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_layers)
        ])
        self.stats_head = StatisticalMatchingHead(d_model)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, x, feats, target_stats=None):
        B, T = x.size()
        x_emb = self.input_proj(x.unsqueeze(-1))
        x_emb = x_emb + self.pos_emb[:, :T, :]
        feat_emb = self.feature_proj(feats)  
        feat_emb = feat_emb.unsqueeze(1) 
        feat_emb = feat_emb * 0.01
        x_emb = x_emb + feat_emb.expand(-1, T, -1)
        for block in self.blocks:
            x_emb = block(x_emb)
        output = self.output_proj(x_emb).squeeze(-1)
        residual_output = torch.sigmoid(self.residual_weight) * x + (1 - torch.sigmoid(self.residual_weight)) * output
        pred_stats = self.stats_head(x_emb)
        
        return residual_output, pred_stats

def statistical_loss(pred, target):
    mse_loss = F.mse_loss(pred, target)
    pred_mean = torch.mean(pred, dim=-1)
    target_mean = torch.mean(target, dim=-1)
    mean_loss = F.mse_loss(pred_mean, target_mean)
    
    pred_std = torch.std(pred, dim=-1)
    target_std = torch.std(target, dim=-1)
    std_loss = F.mse_loss(pred_std, target_std)
    pred_range = torch.max(pred, dim=-1)[0] - torch.min(pred, dim=-1)[0]
    target_range = torch.max(target, dim=-1)[0] - torch.min(target, dim=-1)[0]
    range_loss = F.mse_loss(pred_range, target_range)
    pred_q25 = torch.quantile(pred, 0.25, dim=-1)
    pred_q75 = torch.quantile(pred, 0.75, dim=-1)
    target_q25 = torch.quantile(target, 0.25, dim=-1)
    target_q75 = torch.quantile(target, 0.75, dim=-1)
    
    percentile_loss = F.mse_loss(pred_q25, target_q25) + F.mse_loss(pred_q75, target_q75)
    pred_centered = pred - pred_mean.unsqueeze(-1)
    target_centered = target - target_mean.unsqueeze(-1)

    pred_skew = torch.mean(pred_centered**3, dim=-1) / (pred_std**3 + 1e-8)
    target_skew = torch.mean(target_centered**3, dim=-1) / (target_std**3 + 1e-8)
    skew_loss = F.mse_loss(pred_skew, target_skew)
    pred_kurt = torch.mean(pred_centered**4, dim=-1) / (pred_std**4 + 1e-8)
    target_kurt = torch.mean(target_centered**4, dim=-1) / (target_std**4 + 1e-8)
    kurt_loss = F.mse_loss(pred_kurt, target_kurt)
    
    return {
        'mse': mse_loss,
        'mean': mean_loss,
        'std': std_loss,
        'range': range_loss,
        'percentile': percentile_loss,
        'skew': skew_loss,
        'kurtosis': kurt_loss
    }

def frequency_domain_loss(pred, target):
    pred_fft = torch.fft.fft(pred, dim=-1)
    target_fft = torch.fft.fft(target, dim=-1)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    magnitude_loss = F.mse_loss(pred_mag, target_mag)
    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    phase_loss = F.mse_loss(pred_phase, target_phase)
    pred_psd = pred_mag ** 2
    target_psd = target_mag ** 2
    psd_loss = F.mse_loss(pred_psd, target_psd)
    
    return magnitude_loss + 0.5 * phase_loss + 0.5 * psd_loss


def perfect_fit_training(model, dataloader, feat_mean, feat_std, epochs=200, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    model.train()
    
    for epoch in range(epochs):
        total_losses = {
            'total': 0, 'mse': 0, 'mean': 0, 'std': 0, 'range': 0, 
            'percentile': 0, 'skew': 0, 'kurtosis': 0, 'freq': 0, 'stats': 0
        }
        
        for batch in dataloader:
            seqs, feats = batch
            seqs, feats = seqs.to(device), feats.to(device)
            feats = (feats - feat_mean.to(device)) / feat_std.to(device)
            target_stats = torch.stack([
                torch.mean(seqs, dim=-1),
                torch.std(seqs, dim=-1),
                torch.mean((seqs - torch.mean(seqs, dim=-1, keepdim=True))**3, dim=-1) / (torch.std(seqs, dim=-1)**3 + 1e-8),
                torch.mean((seqs - torch.mean(seqs, dim=-1, keepdim=True))**4, dim=-1) / (torch.std(seqs, dim=-1)**4 + 1e-8),
                torch.max(seqs, dim=-1)[0] - torch.min(seqs, dim=-1)[0]
            ], dim=-1)
            
            optimizer.zero_grad()
        
            output, pred_stats = model(seqs, feats, target_stats)
            stat_losses = statistical_loss(output, seqs)
            freq_loss = frequency_domain_loss(output, seqs)
            stats_loss = F.mse_loss(pred_stats, target_stats)
            total_loss = (
                0.3 * stat_losses['mse'] +
                0.2 * stat_losses['std'] +   
                0.15 * stat_losses['mean'] +
                0.1 * stat_losses['range'] +
                0.1 * stat_losses['percentile'] +
                0.05 * stat_losses['skew'] +
                0.05 * stat_losses['kurtosis'] +
                0.03 * freq_loss +
                0.02 * stats_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            total_losses['total'] += total_loss.item()
            for key, loss in stat_losses.items():
                total_losses[key] += loss.item()
            total_losses['freq'] += freq_loss.item()
            total_losses['stats'] += stats_loss.item()
        
        scheduler.step()
        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Total: {avg_losses['total']:.6f}, MSE: {avg_losses['mse']:.6f}, STD: {avg_losses['std']:.6f}")
        print(f"LR: {current_lr:.2e},Residual Weight: {torch.sigmoid(model.residual_weight).item():.4f}")

def generate_perfect_synthetic_data(model, seeds, feat_mean, feat_std, fs=12000, target_total=4800):
    device = next(model.parameters()).device
    model.eval()
    all_generated = []
    
    with torch.no_grad():
        for seg in seeds:
            feats = extract_combined_features(np.array(seg), fs)
            seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            feat_tensor = torch.tensor(list(feats.values()), dtype=torch.float32).unsqueeze(0).to(device)
            feat_tensor = (feat_tensor - feat_mean.to(device)) / feat_std.to(device)
            pred, _ = model(seg_tensor, feat_tensor)
            pred_np = pred.cpu().numpy().flatten()
            all_generated.extend(pred_np)
            
            if len(all_generated) >= target_total:
                break
    
    return all_generated[:target_total]

if __name__ == "__main__":
    csv_path = "original_data.csv"
    window_size = 256
    raw = load_data_from_csv(csv_path, window_size=window_size)
    dataset = ContinuousSensorDataset(raw, window_size=window_size)

    sample_features = extract_combined_features(np.random.randn(window_size), 12000)
    feature_count = len(sample_features)
    print(f"Feature count: {feature_count} features")
    all_feats = torch.stack([feat for _, feat in dataset])
    feat_mean = all_feats.mean(dim=0)
    feat_std = all_feats.std(dim=0) + 1e-6
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = MultiHead_Model(
        input_len=window_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        dropout=0.3,
        feature_dim=feature_count
    )
    
    print(f"Model Parameters: {count_parameters(model):,}")
    perfect_fit_training(model, dataloader, feat_mean, feat_std, epochs=10, lr=5e-5)
    required_seeds = raw[:2000]
    synthetic_data = generate_perfect_synthetic_data(
        model, required_seeds, feat_mean, feat_std, target_total=48000
    )
    save_data_to_csv(synthetic_data, filename="local-llm/local-llm-data-v6(Ep10).csv")
    
