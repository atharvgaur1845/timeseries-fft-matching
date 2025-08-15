import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import sqrtm
from scipy import signal
import torch.nn.functional as F
import torch.fft as fft
import os
from scipy.stats import skew, kurtosis
import random

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"FORCED GPU USAGE: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - using CPU")
    device = torch.device('cpu')
try:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
except Exception as e:
    pass

win_cache = {}
def get_window(n_fft, device, dtype=torch.float32):
    key = (n_fft, device.type, device.index if device.type=='cuda' else -1, dtype)
    if key not in win_cache:
        win_cache[key] = torch.hann_window(n_fft, device=device, dtype=dtype)
    return win_cache[key]

# XAVIER INITIALIZATION
def init_weights_xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def complex_spectral_loss(pred, target, n_fft=256, hop=None, power=0.3, device=None):
    """
    CRITICAL FIX: Power-compressed magnitude + complex STFT + instantaneous frequency
    Returns dict with mag_loss (power-compressed), complex_l1 (real+imag),
    phase_loss (wrapped), inst_freq_loss (time-derivative of phase).
    """
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
    
    seq_len = pred.size(1)
    n_fft = min(n_fft, seq_len)
    if hop is None:
        hop = max(1, n_fft // 4)
    
    window = get_window(n_fft, pred.device, pred.dtype)

    
    S_pred = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    S_tgt  = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)

    mag_pred = S_pred.abs()
    mag_tgt  = S_tgt.abs()

    comp_pred = (mag_pred + 1e-8) ** power  
    comp_tgt  = (mag_tgt + 1e-8) ** power
    mag_loss = F.l1_loss(comp_pred, comp_tgt)  

    complex_loss = F.l1_loss(S_pred.real, S_tgt.real) + F.l1_loss(S_pred.imag, S_tgt.imag)

    
    phase_pred = torch.angle(S_pred)
    phase_tgt  = torch.angle(S_tgt)
    phase_diff = torch.angle(torch.exp(1j * (phase_pred - phase_tgt))) 
    phase_loss = F.mse_loss(phase_diff, torch.zeros_like(phase_diff))

    if phase_pred.size(-1) > 1:
        dphase_pred = phase_pred[:, :, 1:] - phase_pred[:, :, :-1]
        dphase_tgt  = phase_tgt[:, :, 1:] - phase_tgt[:, :, :-1]
        dphase_pred = torch.angle(torch.exp(1j * dphase_pred))
        dphase_tgt  = torch.angle(torch.exp(1j * dphase_tgt))
        inst_freq_loss = F.mse_loss(dphase_pred, dphase_tgt)
    else:
        inst_freq_loss = torch.tensor(0.0, device=pred.device)
    pred_norm = comp_pred / (comp_pred.sum(dim=(1,2), keepdim=True) + 1e-8)
    tgt_norm  = comp_tgt  / (comp_tgt.sum(dim=(1,2), keepdim=True) + 1e-8)
    shape_loss = F.mse_loss(pred_norm, tgt_norm)

    return {
        'mag': mag_loss,
        'complex': complex_loss,
        'phase': phase_loss,
        'inst_freq': inst_freq_loss,
        'shape': shape_loss
    }

def spectrogram_shape(x, n_fft=256, hop=None, power=0.3):
    """Per-sample normalized spectrogram for shape-only learning"""
    if hop is None:
        hop = max(1, n_fft//4)
    x = x.squeeze(-1)
    seq_len = x.size(1)
    n_fft = min(n_fft, seq_len)
    
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, 
                   window=get_window(n_fft, x.device, x.dtype), return_complex=True)
    mag = (S.abs() + 1e-8) ** power
    mag_norm = mag / (mag.sum(dim=(1,2), keepdim=True) + 1e-8)
    return mag_norm.unsqueeze(1)  # shape (B,1,F,T)

def per_band_mse(real, fake, n_fft=256):
    """Monitor per-frequency-bin MSE to find problematic bands"""
    real = real.squeeze(-1)
    fake = fake.squeeze(-1)
    seq_len = real.size(1)
    n_fft = min(n_fft, seq_len)
    
    S_r = torch.stft(real, n_fft=n_fft, return_complex=True)
    S_f = torch.stft(fake, n_fft=n_fft, return_complex=True)
    mse_per_bin = ((S_r.abs() - S_f.abs())**2).mean(dim=-1).mean(dim=0)  # shape freq_bins
    return mse_per_bin.cpu().numpy()
class PatchSpecDiscriminator(nn.Module):
    """Small patch-based spectrogram discriminator for local texture learning"""
    def __init__(self, n_fft=128):
        super().__init__()
        self.n_fft = n_fft
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),
            nn.Linear(32*8*8, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (B, L, 1)
        x = x.squeeze(-1)
        seq_len = x.size(1)
        n_fft = min(self.n_fft, seq_len)
        hop = max(1, n_fft//4)
        
        S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, 
                       window=get_window(n_fft, x.device, x.dtype), return_complex=True)
        mag = torch.log(S.abs() + 1e-8).unsqueeze(1)  # (B, 1, F, T)
        return self.net(mag)

def normalize_per_sample(x, eps=1e-8):
    """Normalize each sample individually to prevent amplitude cheating"""
    if x.dim() == 3:
        x = x.squeeze(-1)
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=eps)
    x_norm = (x - mu) / std
    amp = torch.cat([mu, std], dim=1)  # shape (B, 2)
    return x_norm.unsqueeze(-1), amp

def apply_amplitude_augmentations(x, aug_prob=0.3):
    """Apply random amplitude scaling augmentation"""
    if random.random() < aug_prob:
        scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, 1, device=x.device)  
        return x * scale
    return x

def apply_time_augmentations(x, aug_prob=0.2): 
    """Apply time domain augmentations"""
    if random.random() < aug_prob:
        max_shift = min(3, x.size(1) // 20)  
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            if shift > 0:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
            else:
                x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
    return x

class RobustCriticTimeFreq(nn.Module):
    """Multi-stream critic with time and frequency domain analysis"""
    def __init__(self, seq_len=128, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.time_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, stride=1),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, padding=2),
            nn.GroupNorm(1, d_model),
        )
        
        self.pos_encoding = LearnablePositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.freq_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveAvgPool1d(1)

        self.fusion_fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(d_model*2 + 32*4*4, d_model)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(d_model, 1))
        )

    def forward(self, x, return_features=False):
        if x.dim() == 3:
            x_t = x.transpose(1,2)
        else:
            x_t = x.unsqueeze(1)

        t = self.time_conv(x_t)
        t = t.transpose(1,2)
        t = self.pos_encoding(t)
        t = self.transformer(t)
        
        t_trans = t.transpose(1,2)
        t_avg = self.global_pool(t_trans).squeeze(-1)
        t_max = self.max_pool(t_trans).squeeze(-1)
        t_pool = torch.cat([t_avg, t_max], dim=1)

        n_fft = min(256, self.seq_len)
        hop = max(1, n_fft // 4)
        x_flat = x.squeeze(-1)
        S = torch.stft(x_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=get_window(n_fft, x.device, x.dtype), return_complex=True)
        mag = S.abs()
        log_mag = torch.log(mag + 1e-8).unsqueeze(1)
        f = self.freq_cnn(log_mag)
        fused = torch.cat([t_pool, f], dim=1)
        out = self.fusion_fc(fused)
        
        if return_features:
            return out, fused
        return out

def sample_patches(x, patch_len=32, num_patches=4):
    """Sample random patches from sequences"""
    B, L, _ = x.shape
    if L < patch_len:
        return x.repeat(num_patches, 1, 1)
    
    patches = []
    for _ in range(num_patches):
        starts = torch.randint(0, L - patch_len + 1, (B,), device=x.device)
        idx = starts.unsqueeze(1) + torch.arange(patch_len, device=x.device).unsqueeze(0)
        p = x.squeeze(-1)[torch.arange(B).unsqueeze(1), idx]
        patches.append(p.unsqueeze(-1))
    return torch.cat(patches, dim=0)

class PatchCritic(nn.Module):
    """Critic operating on short patches"""
    def __init__(self, patch_len=32, d_model=32):
        super().__init__()
        self.patch_len = patch_len
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        else:
            x = x.unsqueeze(1)
        return self.conv_layers(x)

def mrstft_loss(real, fake, n_ffts=(64, 128, 256, 512), hop_factor=0.25):
    """multi-resolution STFT"""
    if real.dim() == 3:
        real = real.squeeze(-1)
        fake = fake.squeeze(-1)
    seq_len = real.size(1)

    total = 0.0
    scales = 0
    for n_fft in n_ffts:
        if n_fft > seq_len:
            continue
        hop = max(1, min(int(n_fft * hop_factor), max(1, seq_len // 4)))
        window = get_window(n_fft, real.device, real.dtype)
        
        real_s = torch.stft(real, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        fake_s = torch.stft(fake, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        
        mag_r = real_s.abs()
        mag_f = fake_s.abs()

        num = torch.linalg.norm((mag_r - mag_f).reshape(mag_r.size(0), -1), dim=1)
        den = torch.linalg.norm(mag_r.reshape(mag_r.size(0), -1), dim=1).clamp_min(1e-8)
        sc = (num / den).mean()
        lm = F.l1_loss(torch.log(mag_f + 1e-8), torch.log(mag_r + 1e-8))

        total = total + sc + lm
        scales += 1

    if scales == 0:
        return torch.tensor(0.0, device=real.device)
    return total / scales

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_len}")
        
        pos_enc = self.pos_embedding[:seq_len].unsqueeze(0)
        pos_enc = pos_enc.expand(batch_size, -1, -1)
        return x + pos_enc

class Generator(nn.Module):
    """Generator for spectral learning with output clipping"""
    def __init__(self, noise_dim=64, seq_len=128, d_model=64, nhead=4, num_layers=6):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.noise_dim = noise_dim

        self.latent_upsampler = nn.Sequential(
            nn.Linear(noise_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, seq_len * d_model),
        )

        self.pos_encoding = LearnablePositionalEncoding(d_model, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(1, d_model)
        )


        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Tanh() )

        self.dc_bias = nn.Parameter(torch.zeros(1))

    def forward(self, noise):
        batch_size = noise.size(0)
        x = self.latent_upsampler(noise)
        x = x.view(batch_size, self.seq_len, self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        output = self.output_projection(x)
        output = output + self.dc_bias
        return output

    def get_dc_regularization(self):
        return 0.01 * self.dc_bias.pow(2).mean()
    
def analyze_batch_spectral(real, fake, save_path=None):
    """Comprehensive spectral """
    real = real.squeeze(-1).cpu().numpy()
    fake = fake.squeeze(-1).cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    
    # Time domain comparison
    plt.subplot(2, 3, 1)
    plt.plot(real[0], label='Real', alpha=0.8)
    plt.plot(fake[0], label='Fake', alpha=0.8)
    plt.title('Time Domain Comparison')
    plt.legend()
    plt.grid(True)
    
    # spectrogram plotting 
    def safe_specgram(x, title, subplot_pos):
        plt.subplot(2, 3, subplot_pos)
        signal_len = len(x)
        
        if signal_len >= 256:
            NFFT = 128
            noverlap = 96
        elif signal_len >= 128:
            NFFT = 64
            noverlap = 48
        elif signal_len >= 64:
            NFFT = 32
            noverlap = 24
        else:
            NFFT = min(16, signal_len)
            noverlap = max(1, int(NFFT * 0.5))
        
        if noverlap >= NFFT:
            noverlap = NFFT - 1
            
        try:
            plt.specgram(x, NFFT=NFFT, noverlap=noverlap, Fs=1, cmap='viridis')
            plt.title(title)
            plt.colorbar()
        except Exception as e:
            plt.plot(np.abs(np.fft.fft(x))[:len(x)//2])
            plt.title(f'{title} (FFT fallback)')
            print(f"Specgram failed for {title}, using FFT: {e}")
    
    safe_specgram(real[0], 'Real Spectrogram', 2)
    safe_specgram(fake[0], 'Fake Spectrogram', 3)
    
    # FFT comparison
    plt.subplot(2, 3, 4)
    real_fft = np.abs(np.fft.fft(real[0]))[:len(real[0])//2]
    fake_fft = np.abs(np.fft.fft(fake[0]))[:len(fake[0])//2]
    freqs = np.fft.fftfreq(len(real[0]))[:len(real[0])//2]
    plt.semilogy(freqs, real_fft, label='Real FFT', alpha=0.8)
    plt.semilogy(freqs, fake_fft, label='Fake FFT', alpha=0.8)
    plt.title('Frequency Spectrum')
    plt.legend()
    plt.grid(True)
    
    # Statistical comparison
    plt.subplot(2, 3, 5)
    real_stats = [np.mean(real[0]), np.std(real[0])]
    fake_stats = [np.mean(fake[0]), np.std(fake[0])]
    x = ['Mean', 'Std']
    width = 0.35
    plt.bar([i - width/2 for i in range(len(x))], real_stats, width, label='Real', alpha=0.7)
    plt.bar([i + width/2 for i in range(len(x))], fake_stats, width, label='Fake', alpha=0.7)
    plt.title('Statistical Comparison')
    plt.ylabel('Value')
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.grid(True)
    
    # Spectral centroids
    plt.subplot(2, 3, 6)
    def compute_centroid(signal):
        fft_vals = np.fft.fft(signal)
        mag = np.abs(fft_vals[:len(signal)//2])
        freqs = np.fft.fftfreq(len(signal))[:len(signal)//2]
        mag_norm = mag / (np.sum(mag) + 1e-8)
        return np.sum(freqs * mag_norm)
    
    real_centroids = [compute_centroid(r) for r in real[:5]]
    fake_centroids = [compute_centroid(f) for f in fake[:5]]
    
    plt.scatter(range(len(real_centroids)), real_centroids, label='Real', alpha=0.7)
    plt.scatter(range(len(fake_centroids)), fake_centroids, label='Fake', alpha=0.7)
    plt.title('Spectral Centroids')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectral analysis saved to {save_path}")
    plt.show()

def gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=10):
    batch_size = real_samples.size(0)  
    alpha = torch.rand(batch_size, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    with torch.backends.cudnn.flags(enabled=False):
        d_interpolates = critic(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def load_data_properly(csv_path, seq_len=128):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 0].values.reshape(-1, 1)
    print(f"Original data shape: {data.shape}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")

    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    normalized_data = scaler.fit_transform(data)

    sequences = []
    stride = seq_len // 2
    for i in range(0, len(normalized_data) - seq_len + 1, stride):
        sequences.append(normalized_data[i:i + seq_len])

    sequences = np.array(sequences)
    print(f"Created {len(sequences)} sequences with 50% overlap, length {seq_len}")
    return torch.FloatTensor(sequences), scaler

def train_wgan_gp(csv_path, output_path='timeseries_synthetic.csv',
                                    epochs=200, batch_size=64, seq_len=128, noise_dim=64,
                                    lr_g=2e-4, lr_c=8e-5, lambda_gp=10, n_critic=5):
    
    print(f"Using device: {device}")

    # Load data
    real_data, scaler = load_data_properly(csv_path, seq_len)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize networks
    generator = Generator(
        noise_dim=noise_dim, seq_len=seq_len, d_model=64, nhead=4, num_layers=6
    ).to(device)

    critic = RobustCriticTimeFreq(
        seq_len=seq_len, d_model=64, nhead=4, num_layers=4
    ).to(device)

    # Patch critic for local patterns
    patch_critic = PatchCritic(patch_len=32, d_model=32).to(device)
    spec_discriminator = PatchSpecDiscriminator(n_fft=128).to(device)
    
    critic.apply(init_weights_xavier)
    patch_critic.apply(init_weights_xavier)
    spec_discriminator.apply(init_weights_xavier)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
    optimizer_pc = optim.Adam(patch_critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
    optimizer_spec = optim.Adam(spec_discriminator.parameters(), lr=lr_c, betas=(0.5, 0.9))


    training_history = {
        'critic_losses': [], 'generator_losses': [], 'patch_critic_losses': [],
        'spec_discriminator_losses': [], 'complex_spectral_losses': [],
        'per_band_mse': []
    }
    
    for epoch in range(epochs):
        epoch_losses = {
            'critic': 0, 'generator': 0, 'patch_critic': 0,
            'spec_discriminator': 0, 'complex_spectral': 0
        }
        num_batches = 0

        
        sigma = max(0.005 * (1 - epoch/(epochs * 0.8)), 0.0)  

        for batch_idx, (real_samples,) in enumerate(dataloader):
            real_samples = real_samples.to(device)
            current_batch_size = real_samples.size(0)

            real_aug = apply_amplitude_augmentations(real_samples, aug_prob=0.2)
            real_aug = apply_time_augmentations(real_aug, aug_prob=0.1)

            real_norm, real_amp = normalize_per_sample(real_samples)
            
            for _ in range(n_critic):
                optimizer_c.zero_grad()
                noise = torch.randn(current_batch_size, noise_dim).to(device)
                fake_samples = generator(noise).detach()
                fake_norm, fake_amp = normalize_per_sample(fake_samples)

                if sigma > 0:
                    real_input = real_norm + sigma * torch.randn_like(real_norm)
                    fake_input = fake_norm + sigma * torch.randn_like(fake_norm)
                else:
                    real_input = real_norm
                    fake_input = fake_norm

                real_output = critic(real_input)
                fake_output = critic(fake_input)

                gp = gradient_penalty(critic, real_norm, fake_norm, device, lambda_gp)
                critic_loss = fake_output.mean() - real_output.mean() + lambda_gp * gp
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer_c.step()

                epoch_losses['critic'] += critic_loss.item()

            optimizer_pc.zero_grad()
            real_patches = sample_patches(real_norm, patch_len=32, num_patches=4)
            fake_patches = sample_patches(fake_norm, patch_len=32, num_patches=4)
            
            real_patch_output = patch_critic(real_patches)
            fake_patch_output = patch_critic(fake_patches)
            
            patch_gp = gradient_penalty(patch_critic, real_patches, fake_patches, device, lambda_gp)
            patch_critic_loss = fake_patch_output.mean() - real_patch_output.mean() + lambda_gp * patch_gp
            patch_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(patch_critic.parameters(), max_norm=1.0)
            optimizer_pc.step()

            epoch_losses['patch_critic'] += patch_critic_loss.item()


            optimizer_spec.zero_grad()
            real_spec_output = spec_discriminator(real_samples)
            fake_spec_output = spec_discriminator(fake_samples.detach())
            
            spec_gp = gradient_penalty(spec_discriminator, real_samples, fake_samples, device, lambda_gp)
            spec_discriminator_loss = fake_spec_output.mean() - real_spec_output.mean() + lambda_gp * spec_gp
            spec_discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(spec_discriminator.parameters(), max_norm=1.0)
            optimizer_spec.step()

            epoch_losses['spec_discriminator'] += spec_discriminator_loss.item()


            optimizer_g.zero_grad()
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_samples = generator(noise)
            fake_norm, fake_amp = normalize_per_sample(fake_samples)

            fake_output, fake_features = critic(fake_norm, return_features=True)
            wasserstein_loss = -fake_output.mean()
            
            fake_patch_output = patch_critic(sample_patches(fake_norm, 32, 4))
            patch_wasserstein_loss = -fake_patch_output.mean()
            
            fake_spec_output = spec_discriminator(fake_samples)
            spec_wasserstein_loss = -fake_spec_output.mean()


            amp_loss = F.l1_loss(fake_amp, real_amp)

            with torch.no_grad():
                _, real_features = critic(real_norm, return_features=True)
            fm_loss = F.l1_loss(fake_features, real_features.detach())


            spec_losses = complex_spectral_loss(
                fake_samples.squeeze(-1), real_samples.squeeze(-1), 
                n_fft=min(256, seq_len), power=0.3 
            )
 
            mrstft_val = mrstft_loss(real_samples, fake_samples, n_ffts=(64,128,256,512))

            dc_reg = generator.get_dc_regularization()

            generator_loss = (
                1.0 * wasserstein_loss +                # Main adversarial
                0.5 * patch_wasserstein_loss +          # Patch adversarial  
                0.8 * spec_wasserstein_loss +           # Spectrogram adversarial
                1.0 * amp_loss +                        # Amplitude matching
                6.0 * fm_loss +                         # UPWEIGHTED: Feature matching
                5.0 * spec_losses['mag'] +              # CRITICAL: Compressed magnitude
                4.0 * spec_losses['shape'] +            # CRITICAL: Shape matching
                2.0 * spec_losses['complex'] +          # Complex (real+imag) matching
                3.0 * spec_losses['inst_freq'] +        # Instantaneous frequency
                5.0 * mrstft_val +                      # UPWEIGHTED: MR-STFT
                dc_reg                                  # DC regularization
            )

            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0) 
            optimizer_g.step()

            # Track losses
            epoch_losses['generator'] += generator_loss.item()
            complex_spectral_total = (spec_losses['mag'] + spec_losses['shape'] + 
                                    spec_losses['complex'] + spec_losses['inst_freq']).item()
            epoch_losses['complex_spectral'] += complex_spectral_total
            num_batches += 1


        for key in epoch_losses:
            if key == 'critic':
                epoch_losses[key] /= max(num_batches * n_critic, 1)
            else:
                epoch_losses[key] /= max(num_batches, 1)


        for key, value in epoch_losses.items():
            training_history[f'{key}_losses'].append(value)

        if epoch % 10 == 0 and epoch > 0:
            generator.eval()
            with torch.no_grad():
                test_noise = torch.randn(32, noise_dim, device=device)
                test_fake = generator(test_noise)
                test_real = real_data[:32].to(device)
                
                band_mse = per_band_mse(test_real, test_fake, n_fft=min(256, seq_len))
                training_history['per_band_mse'].append(band_mse)
                
                # Report worst 5 frequency bins
                worst_bins = np.argsort(band_mse)[-5:]
                print(f"Worst frequency bins: {worst_bins} (MSE: {band_mse[worst_bins]:.4f})")
            generator.train()

        if epoch % 25 == 0 and epoch > 0:
            generator.eval()
            with torch.no_grad():
                test_noise = torch.randn(8, noise_dim, device=device)
                test_fake = generator(test_noise)
                test_real = real_data[:8].to(device)
                
                analyze_batch_spectral(test_real, test_fake, 
                    save_path=f'advanced_spectral_analysis_epoch_{epoch}.png')
            generator.train()

        if epoch % 20 == 0:
            print(f"\nEpoch [{epoch}/{epochs}] - ADVANCED SPECTRAL LEARNING")
            print(f"  Main Critic: {epoch_losses['critic']:.4f}")
            print(f"  Patch Critic: {epoch_losses['patch_critic']:.4f}") 
            print(f"  Spec Discriminator: {epoch_losses['spec_discriminator']:.4f}")
            print(f"  Generator: {epoch_losses['generator']:.4f}")
            print(f"  Complex Spectral: {epoch_losses['complex_spectral']:.4f}")
            print(f"  Instance noise σ: {sigma:.5f}")
    
    generator.eval()
    with torch.no_grad():
        num_samples = len(real_data)
        all_synthetic = []
        
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch, noise_dim).to(device)
            synthetic_batch = generator(noise)
            all_synthetic.append(synthetic_batch)

        synthetic_data = torch.cat(all_synthetic, dim=0)


    print("FINAL ADVANCED SPECTRAL ANALYSIS:")
    analyze_batch_spectral(real_data[:5].to(device), synthetic_data[:5], 
                          save_path='final_advanced_spectral_analysis.png')

    # Save data
    synthetic_flat = synthetic_data.cpu().numpy().reshape(-1, 1)
    denormalized = scaler.inverse_transform(synthetic_flat)
    df = pd.DataFrame(denormalized, columns=['start'])
    df.to_csv(output_path, index=False)
    print(f"synthetic data saved to {output_path}")

    return generator, critic, patch_critic, spec_discriminator, training_history

if __name__ == "__main__":
    csv_file = 'data.csv'
    
    results = train_wgan_gp(
        csv_path=csv_file,
        output_path='synthetic_timeseries.csv',
        epochs=200,
        batch_size=64,
        seq_len=128,
        noise_dim=64,
        lr_g=2e-4,      
        lr_c=8e-5,  
        lambda_gp=10,
        n_critic=5
    )
    

