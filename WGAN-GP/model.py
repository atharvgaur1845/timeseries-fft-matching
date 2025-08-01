import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import math
import os
from scipy.stats import skew, kurtosis, entropy
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
from scipy import linalg

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def compute_time_domain_features(signal):
    eps = 1e-8
    signal = np.array(signal, dtype=np.float64)
    
    mean_val = np.mean(signal)
    std_dev = np.std(signal) + eps
    rms = np.sqrt(np.mean(np.square(signal))) + eps
    abs_mean = np.mean(np.abs(signal)) + eps
    peak_val = np.max(np.abs(signal))
    try:
        skew_val = skew(signal) if std_dev > eps else 0.0
        kurt_val = kurtosis(signal) if std_dev > eps else 0.0
    except:
        skew_val = 0.0
        kurt_val = 0.0
    
    var_val = np.var(signal)
    kurt_index = kurt_val / (rms**4 + eps)
    peak_index = peak_val / (rms + eps)
    waveform_index = rms / (abs_mean + eps)
    pulse_index = peak_val / (abs_mean + eps)
    
    return {
        'mean_value': float(mean_val),
        'standard_deviation': float(std_dev),
        'square_root_amplitude': float(rms),
        'absolute_mean_value': float(abs_mean),
        'peak_value': float(peak_val),
        'skewness': float(skew_val),
        'kurtosis': float(kurt_val),
        'variance': float(var_val),
        'kurtosis_index': float(kurt_index),
        'peak_index': float(peak_index),
        'waveform_index': float(waveform_index),
        'pulse_index': float(pulse_index)
    }

def compute_frequency_domain_features(signal, fs):
    eps = 1e-8
    signal = np.array(signal, dtype=np.float64)
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals)
    fft_freqs = np.fft.fftfreq(len(signal), d=1/fs)
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    spectrum = fft_mag[pos_mask] + eps
    
    norm_spec = spectrum / (np.sum(spectrum) + eps)
    centroid = np.sum(freqs * norm_spec)
    spread = np.sqrt(np.sum((freqs - centroid) ** 2 * norm_spec))
    cumsum_spec = np.cumsum(norm_spec)
    rolloff_indices = np.where(cumsum_spec >= 0.85)[0]
    rolloff = freqs[rolloff_indices[0]] if len(rolloff_indices) > 0 else fs/2
    
    energy = np.sum(spectrum ** 2)
    peak = np.max(spectrum)
    flatness = np.exp(np.mean(np.log(spectrum))) / (np.mean(spectrum) + eps)
    safe_norm_spec = norm_spec + eps
    entropy_val = -np.sum(safe_norm_spec * np.log(safe_norm_spec))
    try:
        freq_skew = skew(spectrum) if np.std(spectrum) > eps else 0.0
        freq_kurt = kurtosis(spectrum) if np.std(spectrum) > eps else 0.0
    except:
        freq_skew = 0.0
        freq_kurt = 0.0
    
    return {
        'frequency_mean_value': float(np.mean(spectrum)),
        'frequency_variance': float(np.var(spectrum)),
        'frequency_skewness': float(freq_skew),
        'frequency_kurtosis': float(freq_kurt),
        'frequency_standard_deviation': float(np.std(spectrum)),
        'frequency_root_mean_square': float(np.sqrt(np.mean(spectrum ** 2))),
        'average_frequency': float(centroid),
        'gravity_frequency': float(centroid),
        'regularity_degree': float(spread),
        'variation_parameter': float(spread / (centroid + eps)),
        'eighth_order_moment': float(np.mean(spectrum ** 8)),
        'sixteenth_order_moment': float(np.mean(spectrum ** 16)),
        'entropy': float(entropy_val),
        'spectral_rolloff_85': float(rolloff),
        'spectral_energy': float(energy),
        'spectral_peak': float(peak),
        'spectral_flatness': float(flatness)
    }

def calculate_frechet_distance(features_real, features_synth):
    eps = 1e-6
    mu_real = np.mean(features_real, axis=0)
    mu_synth = np.mean(features_synth, axis=0)
    sigma_real = np.cov(features_real, rowvar=False) + eps * np.eye(features_real.shape[1])
    sigma_synth = np.cov(features_synth, rowvar=False) + eps * np.eye(features_synth.shape[1])
    
    ssdiff = np.sum((mu_real - mu_synth)**2.0)
    
    try:
        covmean = linalg.sqrtm(sigma_real.dot(sigma_synth))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma_real + sigma_synth - 2.0 * covmean)
    except:
        fid = ssdiff + np.trace(sigma_real) + np.trace(sigma_synth)
    
    return float(fid)

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, embed_dim = query.size()
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_linear(context)
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.2):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_features=False):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        if return_features:
            return x, attn_output
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim=100, seq_len=300, embed_dim=256, num_heads=8, 
                 num_layers=4, ff_dim=512, dropout=0.2):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.noise_projection = nn.Linear(noise_dim, seq_len * embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.deconv = nn.ConvTranspose1d(embed_dim, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        batch_size = noise.size(0)
        x = self.noise_projection(noise)
        x = x.view(batch_size, self.seq_len, self.embed_dim)
        x = x + self.positional_embedding
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.deconv(x)
        x = self.tanh(x)
        x = x.squeeze(1)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.conv = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class Critic(nn.Module):
    def __init__(self, seq_len=300, embed_dim=256, num_heads=8, num_layers=5, 
                 ff_dim=512, dropout=0.3, patch_size=4):
        super(Critic, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.patch_embedding = PatchEmbedding(seq_len, embed_dim, patch_size)
        effective_seq_len = seq_len // patch_size
        
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(torch.randn(1, effective_seq_len + 1, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        x = self.patch_embedding(x)
        
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        x = x + self.positional_encoding
        
        intermediate_features = []
        for i, transformer_block in enumerate(self.transformer_blocks):
            if return_features and i == len(self.transformer_blocks) // 2:
                x, features = transformer_block(x, return_features=True)
                intermediate_features.append(features)
            else:
                x = transformer_block(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        
        if return_features:
            return self.classifier(x).squeeze(-1), intermediate_features
        return self.classifier(x).squeeze(-1)

def calculate_frechet_distance(features_real, features_synth):
    mu_real, sigma_real = np.mean(features_real, axis=0), np.cov(features_real, rowvar=False)
    mu_synth, sigma_synth = np.mean(features_synth, axis=0), np.cov(features_synth, rowvar=False)
    ssdiff = np.sum((mu_real - mu_synth)**2.0)
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_synth), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma_real + sigma_synth - 2.0 * covmean)
    return fid

class GANDataset(Dataset):
    def __init__(self, data, window_size=300, fs=25000):
        self.samples = []
        for segment in data:
            if len(segment) == window_size:
                self.samples.append(segment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32)

class TTSWGANQGP:
    def __init__(self, noise_dim=100, seq_len=300, embed_dim=256, num_heads=8,
                 generator_layers=3, critic_layers=4, ff_dim=512, 
                 generator_dropout=0.2, critic_dropout=0.3,
                 lr=5e-3, lambda_gp=10, lambda_freq=0.1, device='cuda'):
        self.device = device
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.lambda_gp = lambda_gp
        self.lambda_freq = lambda_freq
        
        self.generator = Generator(
            noise_dim=noise_dim,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=generator_layers,
            ff_dim=ff_dim,
            dropout=generator_dropout
        ).to(device)
        
        self.critic = Critic(
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=critic_layers,
            ff_dim=ff_dim,
            dropout=critic_dropout
        ).to(device)
        
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=lr, betas=(0.5, 0.9))
        
        self.fixed_noise = torch.randn(32, noise_dim).to(device)

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = interpolated.requires_grad_(True)
        critic_interpolated = self.critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def frequency_loss(self, real_data, fake_data):
        real_fft = torch.fft.fft(real_data, dim=-1)
        fake_fft = torch.fft.fft(fake_data, dim=-1)
        return F.l1_loss(torch.abs(real_fft), torch.abs(fake_fft))

    def feature_matching_loss(self, real_data, fake_data):
        real_features = []
        fake_features = []
        
        _, real_feat = self.critic(real_data, return_features=True)
        _, fake_feat = self.critic(fake_data, return_features=True)
        
        if real_feat and fake_feat:
            real_features.extend(real_feat)
            fake_features.extend(fake_feat)
            
            fm_loss = 0
            for rf, ff in zip(real_features, fake_features):
                fm_loss += F.l1_loss(torch.mean(rf, dim=0), torch.mean(ff, dim=0))
            return fm_loss / len(real_features)
        return torch.tensor(0.0).to(self.device)

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        
        self.critic.zero_grad()
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise).detach()
        
        real_output = self.critic(real_data)
        fake_output = self.critic(fake_data)
        gp = self.gradient_penalty(real_data, fake_data)
        critic_loss = fake_output.mean() - real_output.mean() + self.lambda_gp * gp
        critic_loss.backward()
        self.optimizer_c.step()
        
        self.generator.zero_grad()
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_output = self.critic(fake_data)
        adversarial_loss = -fake_output.mean()
        freq_loss = self.frequency_loss(real_data, fake_data)
        fm_loss = self.feature_matching_loss(real_data, fake_data)
        
        generator_loss = adversarial_loss + self.lambda_freq * freq_loss + 0.5 * fm_loss
        generator_loss.backward()
        self.optimizer_g.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'generator_loss': generator_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'frequency_loss': freq_loss.item(),
            'feature_matching_loss': fm_loss.item(),
            'gradient_penalty': gp.item()
        }

    def generate_samples(self, num_samples):
        self.generator.eval()
        synthetic_data = []
        
        with torch.no_grad():
            batches = (num_samples + 127) // 128
            for _ in range(batches):
                batch_size = min(128, num_samples - len(synthetic_data))
                if batch_size <= 0:
                    break
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                samples = self.generator(noise)
                synthetic_data.extend(samples.cpu().numpy())
                if len(synthetic_data) >= num_samples:
                    break
        
        self.generator.train()
        return np.array(synthetic_data[:num_samples])

    def compute_fid_score(self, real_data, num_samples=1000):
        synthetic_samples = self.generate_samples(num_samples)
        real_samples = real_data[:num_samples]
        
        real_features = np.array([list(extract_combined_features(np.array(seg), 25000).values()) for seg in real_samples])
        synth_features = np.array([list(extract_combined_features(np.array(seg), 25000).values()) for seg in synthetic_samples])
        
        return calculate_frechet_distance(real_features, synth_features)

def load_data_from_csv(csv_file_path, window_size=300):
    df = pd.read_csv(csv_file_path, header=None)
    raw_sensor_data = []
    for i in range(0, len(df) - window_size + 1, window_size):
        values = df.iloc[i:i+window_size, 0].tolist()
        values = [float(v) for v in values if pd.notna(v)]
        if len(values) == window_size:
            raw_sensor_data.append(values)
    return raw_sensor_data

def save_data_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for v in data:
            f.write(f"{v}\n")

def plot_total_loss_only(loss_history, save_path=None):
    plt.figure(figsize=(14, 8))
    epochs = range(2, len(loss_history['total']) + 1)
    total_loss = loss_history['total'][1:]
    plt.plot(epochs, total_loss, 'b-', linewidth=3, marker='o', markersize=6,
             markerfacecolor='white', markeredgecolor='blue', markeredgewidth=2,
             label='Total Loss', alpha=0.9)
    plt.title('Training Loss Analysis (Excluding Epoch 1)', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Total Loss', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.6, linestyle='--', linewidth=1)
    min_loss = min(total_loss)
    max_loss = max(total_loss)
    min_epoch = total_loss.index(min_loss) + 2
    max_epoch = total_loss.index(max_loss) + 2
    final_loss = total_loss[-1]
    plt.scatter(min_epoch, min_loss, color='red', s=150, zorder=5, marker='*', label='Minimum Loss', edgecolor='darkred', linewidth=2)
    plt.scatter(max_epoch, max_loss, color='orange', s=120, zorder=5, marker='^', label='Maximum Loss', edgecolor='darkorange', linewidth=2)
    plt.axhline(y=min_loss, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.axvline(x=min_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    improvement_pct = ((max_loss - min_loss) / max_loss) * 100 if max_loss > 0 else 0
    plt.text(0.02, 0.98, f'Loss Reduction: {improvement_pct:.2f}%',
            transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8), verticalalignment='top')
    
    last_5_epochs = total_loss[-5:] if len(total_loss) >= 5 else total_loss
    convergence_std = np.std(last_5_epochs)
    convergence_status = "Converged" if convergence_std < 0.001 else "Still Learning"
    plt.text(0.02, 0.88, f'Convergence: {convergence_status}\nStd (last 5): {convergence_std:.6f}',
            transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.8), verticalalignment='top')
    plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    y_range = max_loss - min_loss
    plt.ylim(min_loss - y_range*0.1, max_loss + y_range*0.2)
    plt.xlim(1.5, len(epochs) + 1.5)
    plt.gca().set_facecolor('#f8f9fa')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"total loss plot saved to: {save_path}")
    plt.show()

def plot_sample_waveforms(real_data, synthetic_data, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(real_data[:300], 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Real Signal Sample', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(synthetic_data[:300], 'r-', linewidth=1, alpha=0.8)
    axes[1].set_title('Generated Signal Sample', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"sample waveforms saved to: {save_path}")
    plt.show()

def advanced_training(model, dataloader, real_data_for_fid, epochs=200, window_size=300, fs=25000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.generator.to(device)
    model.critic.to(device)
    
    scheduler_g = torch.optim.lr_scheduler.StepLR(model.optimizer_g, step_size=50, gamma=0.8)
    scheduler_c = torch.optim.lr_scheduler.StepLR(model.optimizer_c, step_size=50, gamma=0.8)
    
    loss_history = {
        'total': [], 'critic_loss': [], 'generator_loss': [], 'adversarial_loss': [],
        'frequency_loss': [], 'gradient_penalty': [], 'feature_matching_loss': [], 'fid_scores': []
    }
    
    best_fid = float('inf')
    best_generator_state = None
    
    for epoch in range(epochs):
        epoch_losses = {k: [] for k in loss_history.keys() if k != 'fid_scores'}
        
        for batch in dataloader:
            seqs = batch.to(device)
            losses = model.train_step(seqs)
            
            total_loss = losses['generator_loss'] + losses['critic_loss']
            epoch_losses['total'].append(total_loss)
            
            for key in ['critic_loss', 'generator_loss', 'adversarial_loss', 
                       'frequency_loss', 'gradient_penalty', 'feature_matching_loss']:
                epoch_losses[key].append(losses[key])
        
        scheduler_g.step()
        scheduler_c.step()
        
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        for key, value in avg_losses.items():
            loss_history[key].append(value)
        
        if (epoch + 1) % 10 == 0:
            fid_score = model.compute_fid_score(real_data_for_fid, num_samples=500)
            loss_history['fid_scores'].append(fid_score)
            
            if fid_score < best_fid:
                best_fid = fid_score
                best_generator_state = model.generator.state_dict().copy()
            
            synthetic_sample = model.generate_samples(1)[0]
            real_sample = real_data_for_fid[0]
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {avg_losses['total']:.6f}")
            print(f"  Generator: {avg_losses['generator_loss']:.6f} | Critic: {avg_losses['critic_loss']:.6f}")
            print(f"  Adversarial: {avg_losses['adversarial_loss']:.6f} | Freq: {avg_losses['frequency_loss']:.6f}")
            print(f"  Feature Matching: {avg_losses['feature_matching_loss']:.6f} | FID: {fid_score:.6f}")
            print(f"  Best FID: {best_fid:.6f}")
            print("-" * 80)
    
    if best_generator_state is not None:
        model.generator.load_state_dict(best_generator_state)
    
    return loss_history

def train_model_with_csv(csv_file_path, output_dir="WGAN-GP/synthetic_data", epochs=200, batch_size=128, window_size=300, fs=25000):
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
    raw_data = load_data_from_csv(csv_file_path, window_size=window_size)
    dataset = GANDataset(raw_data, window_size=window_size, fs=fs)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = TTSWGANQGP(
        seq_len=window_size,
        device=device
    )
    
    print(f"Generator Parameters: {count_parameters(model.generator):,}")
    print(f"Critic Parameters: {count_parameters(model.critic):,}")
    
    data_min = np.min([np.min(seg) for seg in raw_data])
    data_max = np.max([np.max(seg) for seg in raw_data])
    raw_data_normalized = [2 * (np.array(seg) - data_min) / (data_max - data_min) - 1 for seg in raw_data]
    
    print(f"training for {epochs} epochs ")
    
    loss_history = advanced_training(
        model, dataloader, raw_data_normalized,
        epochs=epochs, window_size=window_size, fs=fs
    )
    
    plot_total_loss_only(loss_history, save_path=os.path.join(output_dir, "total_loss.png"))
    target_total = len(raw_data) * window_size
    synthetic_samples = model.generate_samples(target_total)
    synthetic_data_flat = synthetic_samples.flatten()
    synthetic_data_denormalized = (synthetic_data_flat + 1) * (data_max - data_min) / 2 + data_min
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "synthetic_data.csv")
    save_data_to_csv(synthetic_data_denormalized, output_file)
    
    original_flat = np.concatenate(raw_data)
    plot_sample_waveforms(original_flat, synthetic_data_denormalized, 
                         save_path=os.path.join(output_dir, "sample_comparison.png"))
    
    return model, synthetic_data_denormalized

if __name__ == "__main__":
    csv_file_path = "data.csv"
    window_size = 300
    
    trained_model, synthetic_data = train_model_with_csv(
        csv_file_path=csv_file_path,
        output_dir="WGAN-GP/synthetic_data",
        epochs=200,
        batch_size=128,
        window_size=window_size,
        fs=25000
    )

