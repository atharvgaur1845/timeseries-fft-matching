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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# FIX 1: Safe CUDA backend wrapper
try:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
except Exception as e:
    # ignore if flags not available in this build
    pass

# CRITICAL BUG FIX 3: Robust window cache with proper device/dtype handling
win_cache = {}
def get_window(n_fft, device, dtype=torch.float32):
    """Precomputed Hann windows with robust device/dtype handling"""
    key = (n_fft, device.type, device.index if device.type=='cuda' else -1, dtype)
    if key not in win_cache:
        win_cache[key] = torch.hann_window(n_fft, device=device, dtype=dtype)
    return win_cache[key]

# FIX 2: Time-domain losses
def time_domain_l1_loss(real, fake):
    """Direct waveform L1 loss for envelope/timing matching"""
    if real.dim() == 3:
        real = real.squeeze(-1)
        fake = fake.squeeze(-1)
    return F.l1_loss(fake, real)

def derivative_l1_loss(real, fake):
    """First-order derivative loss for temporal smoothness"""
    if real.dim() == 3:
        real = real.squeeze(-1)
        fake = fake.squeeze(-1)
    dr = real[:, 1:] - real[:, :-1]
    df = fake[:, 1:] - fake[:, :-1]
    return F.l1_loss(df, dr)

# CRITICAL BUG FIX 2: PSD loss with hop=0 protection
def batch_log_psd(x, n_fft=256, hop=128):
    """Compute log PSD using Welch-like method with proper n_fft sizing and hop protection"""
    if x.dim() == 3:
        x = x.squeeze(-1)
    
    seq_len = x.size(1)
    # CRITICAL FIX: Ensure n_fft doesn't exceed seq_len
    n_fft = min(n_fft, seq_len)
    # CRITICAL BUG FIX 2: Ensure hop at least 1 and not larger than quarter length
    hop = max(1, min(hop, max(1, seq_len // 4)))
    
    window = get_window(n_fft, x.device, x.dtype)  # Use robust cached window
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    P = (S.abs() ** 2).mean(dim=-1)   # (B, F)
    return torch.log(P + 1e-8)

def psd_loss(real, fake, n_fft=256, hop=128):
    """Power spectral density matching loss"""
    pr = batch_log_psd(real, n_fft=n_fft, hop=hop)
    pf = batch_log_psd(fake, n_fft=n_fft, hop=hop)
    return F.mse_loss(pf, pr)

# FIX 2: Autocorrelation loss with simplified nfft calculation
def autocorr(x):
    """Returns autocorr for lags 0..L-1, x: (B,L) or (B,L,1)"""
    if x.dim() == 3:
        x = x.squeeze(-1)
    n = x.shape[-1]
    # CRITICAL FIX: Simplified power-of-2 calculation
    nfft = 1 << int(np.ceil(np.log2(n)))  # Fast, clear power-of-2 >= n
    X = fft.rfft(x, n=nfft)
    S = (X * torch.conj(X)).real
    ac = fft.irfft(S, n=nfft)
    ac = ac[:, :n]
    return ac

def autocorr_loss(real, fake, max_lag=128):
    """Autocorrelation matching for periodic patterns"""
    ar = autocorr(real)[:, :max_lag]
    af = autocorr(fake)[:, :max_lag]
    # CRITICAL FIX: Better numerical stability
    ar = ar / (ar[:, :1].clamp(min=1e-8))
    af = af / (af[:, :1].clamp(min=1e-8))
    return F.mse_loss(af, ar)

# FIX 3: Improved Multi-Resolution STFT Loss with cached windows and hop protection
def mrstft_loss(real, fake, n_ffts=(64, 128, 256), hop_factor=0.25):
    """Multi-resolution STFT: spectral-convergence + log-mag L1 averaged across valid scales."""
    # Accept shapes (B,L,1) or (B,L)
    if real.dim() == 3:
        real = real.squeeze(-1)
        fake = fake.squeeze(-1)
    seq_len = real.size(1)

    total = 0.0
    scales = 0
    for n_fft in n_ffts:
        # CRITICAL FIX: Ensure n_fft doesn't exceed seq_len
        if n_fft > seq_len:
            continue
        # CRITICAL BUG FIX 2: Consistent hop calculation with protection
        hop = max(1, min(int(n_fft * hop_factor), max(1, seq_len // 4)))
        window = get_window(n_fft, real.device, real.dtype)  # Use robust cached window
        real_s = torch.stft(real, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        fake_s = torch.stft(fake, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        mag_r = real_s.abs()
        mag_f = fake_s.abs()

        # spectral convergence (Frobenius)
        num = torch.linalg.norm((mag_r - mag_f).reshape(mag_r.size(0), -1), dim=1)
        den = torch.linalg.norm(mag_r.reshape(mag_r.size(0), -1), dim=1).clamp_min(1e-8)
        sc = (num / den).mean()

        # log mag l1
        lm = F.l1_loss(torch.log(mag_f + 1e-8), torch.log(mag_r + 1e-8))

        total = total + sc + lm
        scales += 1

    if scales == 0:
        return torch.tensor(0.0, device=real.device)
    return total / scales

# --- Torch-based spectral feature extractor with window caching ---
def extract_torch_spectral_features(x, n_fft=None, hop_length=None, device=None):
    """Fast spectral feature extraction with cached windows"""
    if device is None:
        device = x.device
    
    x = x.squeeze(-1) # (B, seq)
    seq_len = x.size(1)
    
    # Auto-adjust n_fft with proper sizing
    if n_fft is None or n_fft > seq_len:
        n_fft = min(256, seq_len // 2 * 2)
    if hop_length is None:
        hop_length = n_fft // 4
    
    n_fft = min(n_fft, seq_len)
    hop_length = max(1, min(hop_length, max(1, seq_len // 4)))  # Protection against hop=0
    
    # IMPROVEMENT: Use cached window
    window = get_window(n_fft, device, x.dtype)
    stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    mag = torch.abs(stft) # (B, F, T)
    
    # Summary features
    mag_mean = mag.mean(dim=-1) # (B, F)
    mag_std = mag.std(dim=-1) # (B, F)
    
    # Spectral centroid and bandwidth
    freqs = torch.linspace(0, 0.5, mag.size(1), device=device)
    centroid = (mag_mean * freqs).sum(dim=1) / (mag_mean.sum(dim=1) + 1e-8)
    bandwidth = torch.sqrt(((freqs - centroid.unsqueeze(1))**2 * mag_mean).sum(dim=1) / (mag_mean.sum(dim=1) + 1e-8))
    
    features = torch.cat([mag_mean, mag_std, centroid.unsqueeze(1), bandwidth.unsqueeze(1)], dim=1)
    return features

# --- Instance noise for stability ---
def add_instance_noise(x, sigma):
    """Add instance noise that decays during training"""
    if sigma <= 0:
        return x
    return x + (sigma * torch.randn_like(x))

class SimpleLearnablePositionalEncoding(nn.Module):
    """Fixed learnable positional encoding"""
    def __init__(self, d_model, max_len=1000):
        super(SimpleLearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_len}")
        
        pos_enc = self.pos_embedding[:seq_len].unsqueeze(0)
        pos_enc = pos_enc.expand(batch_size, -1, -1)
        return x + pos_enc

class FixedImprovedGenerator(nn.Module):
    """Generator with all critical fixes applied"""
    def __init__(self, noise_dim=64, seq_len=128, d_model=64, nhead=4, num_layers=6):
        super(FixedImprovedGenerator, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.noise_dim = noise_dim

        # FIX 4: Remove Tanh from latent upsampler
        self.latent_upsampler = nn.Sequential(
            nn.Linear(noise_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, seq_len * d_model),
            # no Tanh! leave it linear so transformer can shape it
        )

        # Positional encoding
        self.pos_encoding = SimpleLearnablePositionalEncoding(d_model, seq_len)

        # Transformer layers - FIX 5: Smaller by default
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Fixed temporal convolution with GroupNorm
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(1, d_model)
        )

        # CRITICAL FIX: Remove final Tanh for better distribution matching
        self.output_projection = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(d_model, d_model // 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.utils.spectral_norm(nn.Linear(d_model // 2, 1)),
            # REMOVED nn.Tanh() - rely on loss regularization instead
        )

        # DC bias with regularization
        self.dc_bias = nn.Parameter(torch.zeros(1))

    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Upsample single latent vector to sequence
        x = self.latent_upsampler(noise) # (B, seq_len * d_model)
        x = x.view(batch_size, self.seq_len, self.d_model) # (B, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply temporal convolution
        x_conv = x.transpose(1, 2) # (B, D, L)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2) # (B, L, D)
        
        # Residual connection
        x = x + x_conv
        
        # Output projection (no Tanh!)
        output = self.output_projection(x)
        output = output + self.dc_bias
        
        return output

    def get_dc_regularization(self):
        """DC bias regularization"""
        return 0.01 * self.dc_bias.pow(2).mean()

class FixedImprovedCritic(nn.Module):
    """Critic with all critical fixes applied"""
    def __init__(self, seq_len=128, d_model=64, nhead=4, num_layers=6):
        super(FixedImprovedCritic, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection with spectral normalization
        self.input_projection = nn.utils.spectral_norm(
            nn.Linear(1, d_model)
        )

        # Positional encoding
        self.pos_encoding = SimpleLearnablePositionalEncoding(d_model, seq_len)

        # Transformer layers - FIX 5: Smaller by default
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Fixed temporal convolution with GroupNorm
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GroupNorm(1, d_model)
        )

        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Output classifier
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(d_model * 2, d_model)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(d_model, 1))
        )

    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply temporal convolution
        x_conv = x.transpose(1, 2) # (B, D, L)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2) # (B, L, D)
        
        # Residual connection
        x = x + x_conv
        
        # Multi-scale pooling
        x = x.transpose(1, 2) # (B, D, L) for pooling
        avg_pooled = self.global_pool(x).squeeze(-1)
        max_pooled = self.max_pool(x).squeeze(-1)
        combined = torch.cat([avg_pooled, max_pooled], dim=1)
        
        output = self.classifier(combined)
        return output

class FastFeatureExtractor(nn.Module):
    """Fast pytorch-only feature extractor for FID"""
    def __init__(self, seq_len=128):
        super(FastFeatureExtractor, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.feature_net(x)
        return features

def improved_gradient_penalty(critic, real_samples, fake_samples, device):
    """Improved gradient penalty with CuDNN disabled"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Disable CuDNN for gradient computation
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

def load_data_properly_fixed(csv_path, seq_len=128):
    """Proper 50% overlap implementation"""
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 0].values.reshape(-1, 1)
    print(f"Original data shape: {data.shape}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")

    # CRITICAL FIX: Adjust scaler range to match generator output (no Tanh)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))  # Wider range since no Tanh
    normalized_data = scaler.fit_transform(data)

    # Proper 50% overlap
    sequences = []
    stride = seq_len // 2 # 50% overlap
    for i in range(0, len(normalized_data) - seq_len + 1, stride):
        sequences.append(normalized_data[i:i + seq_len])

    sequences = np.array(sequences)
    print(f"Created {len(sequences)} sequences with 50% overlap, length {seq_len}")
    return torch.FloatTensor(sequences), scaler

def compute_psd_metrics(real_data, fake_data, n_fft=256):
    """Compute PSD comparison metrics for debugging"""
    # CRITICAL FIX: Ensure proper n_fft sizing
    seq_len = real_data.size(1)
    n_fft = min(n_fft, seq_len)
    
    real_psd = batch_log_psd(real_data, n_fft=n_fft).mean(dim=0)
    fake_psd = batch_log_psd(fake_data, n_fft=n_fft).mean(dim=0)
    mse_psd = F.mse_loss(fake_psd, real_psd).item()
    return mse_psd, real_psd, fake_psd

def train_fixed_wgan_gp(csv_path, output_path='FIXED_synthetic_timeseries.csv',
                       epochs=200, batch_size=64, seq_len=128, noise_dim=64,
                       lr_g=1e-4, lr_c=2e-4, lambda_gp=10, n_critic=5,
                       # CRITICAL FIX: Loss balancing experiments
                       lambda_adv=1.0, lambda_time=5.0, lambda_deriv=0.5, 
                       lambda_mrstft=1.0, lambda_feat=1.0, lambda_stat=0.1, 
                       lambda_psd=0.2, lambda_acorr=0.2):
    """Train WGAN-GP with ALL critical fixes applied"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    real_data, scaler = load_data_properly_fixed(csv_path, seq_len)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Data loaded - sequences: {len(real_data)}, sequence length: {seq_len}")

    # FIX 5: Initialize networks with smaller defaults
    generator = FixedImprovedGenerator(
        noise_dim=noise_dim,
        seq_len=seq_len,
        d_model=64,        # was 128
        nhead=4,
        num_layers=6       # was 12
    ).to(device)

    critic = FixedImprovedCritic(
        seq_len=seq_len,
        d_model=64,        # was 128
        nhead=4,
        num_layers=6       # was 14
    ).to(device)

    feature_extractor = FastFeatureExtractor(seq_len).to(device)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

    # CRITICAL BUG FIX 1: Create full evaluation noise for dataset-length FID eval
    dataset_len = len(dataloader.dataset)
    torch.manual_seed(42)  # Deterministic eval noise for reproducibility
    eval_noise_full = torch.randn(dataset_len, noise_dim, device=device)
    eval_noise_psd = torch.randn(200, noise_dim, device=device)  # For PSD evaluation
    print(f"✓ Full evaluation noise created: {eval_noise_full.shape}")

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=lr_c, betas=(0.0, 0.9))

    # Schedulers
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=epochs//4, gamma=0.5)
    scheduler_c = optim.lr_scheduler.StepLR(optimizer_c, step_size=epochs//4, gamma=0.5)

    # Training history - FIX 6: Track all losses
    critic_losses = []
    generator_losses = []
    mrstft_losses = []
    time_l1_losses = []
    deriv_l1_losses = []
    psd_losses = []
    acorr_losses = []
    feat_losses = []
    stat_losses = []
    dc_reg_losses = []
    fid_scores = []
    psd_metrics = []
    
    best_fid_score = float('inf')
    best_model_path = 'best_fixed_wgan_gp.pth'

    print("Starting training with ALL CRITICAL BUG FIXES...")
    print(f"Loss weights: adv={lambda_adv}, time={lambda_time}, deriv={lambda_deriv}, mrstft={lambda_mrstft}")
    
    for epoch in range(epochs):
        epoch_critic_loss = 0
        epoch_generator_loss = 0
        epoch_mrstft_loss = 0
        epoch_time_l1_loss = 0
        epoch_deriv_l1_loss = 0
        epoch_psd_loss = 0
        epoch_acorr_loss = 0
        epoch_feat_loss = 0
        epoch_stat_loss = 0
        epoch_dc_reg = 0
        num_batches = 0

        # Instance noise schedule (slightly increased initial value)
        sigma = max(0.1 * (1 - epoch/(epochs * 1.2)), 0.001)

        for batch_idx, (real_samples,) in enumerate(dataloader):
            real_samples = real_samples.to(device)
            current_batch_size = real_samples.size(0)

            # Train Critic
            for _ in range(n_critic):
                optimizer_c.zero_grad()
                noise = torch.randn(current_batch_size, noise_dim).to(device)
                fake_samples = generator(noise).detach()

                # Add instance noise
                real_noisy = add_instance_noise(real_samples, sigma)
                fake_noisy = add_instance_noise(fake_samples, sigma)

                # Critic outputs
                real_output = critic(real_noisy)
                fake_output = critic(fake_noisy)

                # Gradient penalty
                gp = improved_gradient_penalty(critic, real_samples, fake_samples, device)

                # Critic loss
                critic_loss = fake_output.mean() - real_output.mean() + lambda_gp * gp
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer_c.step()

                epoch_critic_loss += critic_loss.item()

            # FIX 6: Train Generator with ALL new losses
            optimizer_g.zero_grad()
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_samples = generator(noise)
            fake_output = critic(fake_samples)

            # Adversarial loss
            wasserstein_loss = -fake_output.mean()

            # FIX 3: MR-STFT with improved scales and consistent sizing
            mrstft_val = mrstft_loss(real_samples, fake_samples, n_ffts=(64,128,256))

            # FIX 2: Time-domain L1 + derivative
            time_l1 = time_domain_l1_loss(real_samples, fake_samples)
            deriv_l1 = derivative_l1_loss(real_samples, fake_samples)

            # FIX 2: PSD + autocorr with proper sizing
            psd_l = psd_loss(real_samples, fake_samples, n_fft=min(512, seq_len), hop=max(32, seq_len//8))
            acor_l = autocorr_loss(real_samples, fake_samples, max_lag=min(128, seq_len//2))

            # Spectral feature matching
            feat_real = extract_torch_spectral_features(real_samples, device=device)
            feat_fake = extract_torch_spectral_features(fake_samples, device=device)
            feat_loss = F.l1_loss(feat_fake, feat_real)

            # Mean/std matching
            mean_loss = F.l1_loss(fake_samples.mean(dim=[1,2]), real_samples.mean(dim=[1,2]))
            std_loss = F.l1_loss(fake_samples.std(dim=[1,2]), real_samples.std(dim=[1,2]))
            stat_loss = mean_loss + std_loss

            # DC reg
            dc_reg = generator.get_dc_regularization()

            # FIX 6: Combined generator loss with ALL fixes
            generator_loss = (
                lambda_adv * wasserstein_loss +
                lambda_time * time_l1 +
                lambda_deriv * deriv_l1 +
                lambda_mrstft * mrstft_val +
                lambda_feat * feat_loss +
                lambda_stat * stat_loss +
                lambda_psd * psd_l +
                lambda_acorr * acor_l +
                dc_reg
            )

            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()

            # FIX 6: Track FULL generator loss and all components
            epoch_generator_loss += generator_loss.item()
            epoch_mrstft_loss += mrstft_val.item()
            epoch_time_l1_loss += time_l1.item()
            epoch_deriv_l1_loss += deriv_l1.item()
            epoch_psd_loss += psd_l.item()
            epoch_acorr_loss += acor_l.item()
            epoch_feat_loss += feat_loss.item()
            epoch_stat_loss += stat_loss.item()
            epoch_dc_reg += dc_reg.item()
            num_batches += 1

        # Update learning rates
        scheduler_g.step()
        scheduler_c.step()

        # Calculate averages with protection against divide-by-zero
        avg_critic_loss = epoch_critic_loss / max(num_batches * n_critic, 1)
        avg_generator_loss = epoch_generator_loss / max(num_batches, 1)
        avg_mrstft_loss = epoch_mrstft_loss / max(num_batches, 1)
        avg_time_l1_loss = epoch_time_l1_loss / max(num_batches, 1)
        avg_deriv_l1_loss = epoch_deriv_l1_loss / max(num_batches, 1)
        avg_psd_loss = epoch_psd_loss / max(num_batches, 1)
        avg_acorr_loss = epoch_acorr_loss / max(num_batches, 1)
        avg_feat_loss = epoch_feat_loss / max(num_batches, 1)
        avg_stat_loss = epoch_stat_loss / max(num_batches, 1)
        avg_dc_reg = epoch_dc_reg / max(num_batches, 1)

        # Store losses
        critic_losses.append(avg_critic_loss)
        generator_losses.append(avg_generator_loss)
        mrstft_losses.append(avg_mrstft_loss)
        time_l1_losses.append(avg_time_l1_loss)
        deriv_l1_losses.append(avg_deriv_l1_loss)
        psd_losses.append(avg_psd_loss)
        acorr_losses.append(avg_acorr_loss)
        feat_losses.append(avg_feat_loss)
        stat_losses.append(avg_stat_loss)
        dc_reg_losses.append(avg_dc_reg)

        # FIX 7: More frequent PSD evaluation with FIXED noise
        if epoch % 10 == 0 and epoch > 0:
            generator.eval()
            with torch.no_grad():
                # Use fixed evaluation noise for PSD
                fake_batch = generator(eval_noise_psd)
                real_batch = real_data[:200].to(device)
                
                mse_psd, _, _ = compute_psd_metrics(real_batch, fake_batch)
                psd_metrics.append(mse_psd)
                
                print(f"Epoch {epoch}: PSD MSE = {mse_psd:.4f} (FIXED NOISE)")
            generator.train()

        # CRITICAL BUG FIX 1: Evaluate FID with proper noise indexing
        if epoch % 50 == 0 and epoch > 0:  # Changed from 100 to 50
            print(f"Evaluating FID at epoch {epoch}...")
            generator.eval()
            feature_extractor.eval()
            
            real_features_list = []
            fake_features_list = []
            
            with torch.no_grad():
                idx = 0  # Reset index counter for proper noise slicing
                for batch_data, in dataloader:
                    batch_data = batch_data.to(device)
                    current_batch_size = batch_data.size(0)
                    
                    # Real features
                    real_feat = feature_extractor(batch_data)
                    real_features_list.append(real_feat)
                    
                    # CRITICAL BUG FIX 1: Use proper noise indexing to avoid repeats
                    start_idx = idx
                    end_idx = idx + current_batch_size
                    noise_batch = eval_noise_full[start_idx:end_idx]
                    idx += current_batch_size
                    
                    fake_data = generator(noise_batch)
                    fake_feat = feature_extractor(fake_data)
                    fake_features_list.append(fake_feat)

                real_features = torch.cat(real_features_list, dim=0)
                fake_features = torch.cat(fake_features_list, dim=0)

                # Calculate FID
                real_features_np = real_features.cpu().numpy()
                fake_features_np = fake_features.cpu().numpy()
                
                mu1, sigma1 = real_features_np.mean(axis=0), np.cov(real_features_np, rowvar=False)
                mu2, sigma2 = fake_features_np.mean(axis=0), np.cov(fake_features_np, rowvar=False)
                
                # Numerical stability
                eps = 1e-6
                sigma1 += eps * np.eye(sigma1.shape[0])
                sigma2 += eps * np.eye(sigma2.shape[0])
                
                ssdiff = np.sum((mu1 - mu2) ** 2.0)
                covmean = sqrtm(sigma1.dot(sigma2))
                
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                
                fid_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
                fid_scores.append(fid_score)

                if fid_score < best_fid_score:
                    best_fid_score = fid_score
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': generator.state_dict(),
                        'critic_state_dict': critic.state_dict(),
                        'fid_score': fid_score
                    }, best_model_path)
                    print(f"✓ New best model saved! FID: {fid_score:.4f} (NO REPEATS)")

            generator.train()
            feature_extractor.train()

        # FIX 7: Enhanced logging every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{epochs}]")
            print(f"  Critic: {avg_critic_loss:.4f} | Total Gen: {avg_generator_loss:.4f}")
            print(f"  Time L1: {avg_time_l1_loss:.4f} | Deriv L1: {avg_deriv_l1_loss:.4f}")
            print(f"  MR-STFT: {avg_mrstft_loss:.4f} | PSD: {avg_psd_loss:.4f}")
            print(f"  Autocorr: {avg_acorr_loss:.4f} | DC Reg: {avg_dc_reg:.6f}")
            print(f"  Noise σ: {sigma:.4f} | Time/STFT ratio: {avg_time_l1_loss/max(avg_mrstft_loss, 1e-8):.2f}")

    # CRITICAL FIX: Conditional FID printing
    if fid_scores:
        print(f"Training completed! Best FID: {best_fid_score:.4f}")
    else:
        print("Training completed! No FID computed.")

    # Load best model and generate final data
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print("Loaded best model for final generation")

    # Generate synthetic data
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

    # Save synthetic data
    synthetic_flat = synthetic_data.cpu().numpy().reshape(-1, 1)
    denormalized = scaler.inverse_transform(synthetic_flat)
    df = pd.DataFrame(denormalized, columns=['fixed_synthetic_timeseries'])
    df.to_csv(output_path, index=False)
    print(f"Fixed synthetic data saved to {output_path}")

    # Create comprehensive analysis plots
    plt.figure(figsize=(20, 16))

    # Loss plots with all new losses
    plt.subplot(4, 4, 1)
    plt.plot(critic_losses, label='Critic Loss', alpha=0.7)
    plt.title('Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 2)
    plt.plot(generator_losses, label='Total Generator Loss', alpha=0.7, color='red')
    plt.title('Total Generator Loss (BUG-FIXED)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 3)
    plt.plot(time_l1_losses, label='Time-domain L1', alpha=0.7, color='blue')
    plt.title(f'Time-Domain L1 Loss (λ={lambda_time})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 4)
    plt.plot(deriv_l1_losses, label='Derivative L1', alpha=0.7, color='green')
    plt.title(f'Derivative L1 Loss (λ={lambda_deriv})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 5)
    plt.plot(mrstft_losses, label='MR-STFT Loss', alpha=0.7, color='orange')
    plt.title(f'Multi-Resolution STFT Loss (λ={lambda_mrstft})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 6)
    plt.plot(psd_losses, label='PSD Loss', alpha=0.7, color='purple')
    plt.title(f'PSD Loss (λ={lambda_psd}) - hop≥1 fixed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 7)
    plt.plot(acorr_losses, label='Autocorr Loss', alpha=0.7, color='brown')
    plt.title(f'Autocorrelation Loss (λ={lambda_acorr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 8)
    plt.plot(dc_reg_losses, label='DC Regularization', alpha=0.7, color='pink')
    plt.title('DC Bias Regularization')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # PSD metrics evolution with FIXED noise
    plt.subplot(4, 4, 9)
    if len(psd_metrics) > 0:
        psd_epochs = list(range(10, 10 * len(psd_metrics) + 1, 10))
        plt.plot(psd_epochs, psd_metrics, 'go-', alpha=0.7)
        plt.title('PSD MSE Evolution (FIXED NOISE)')
        plt.xlabel('Epoch')
        plt.ylabel('PSD MSE')
        plt.grid(True)

    # FID scores with proper noise indexing
    plt.subplot(4, 4, 10)
    if len(fid_scores) > 0:
        fid_epochs = list(range(50, 50 * len(fid_scores) + 1, 50))
        plt.plot(fid_epochs, fid_scores, 'bo-', alpha=0.7)
        if best_fid_score != float('inf'):
            plt.axhline(y=best_fid_score, color='red', linestyle='--',
                       label=f'Best: {best_fid_score:.4f}')
        plt.title('FID Score (NO REPEATS)')
        plt.xlabel('Epoch')
        plt.ylabel('FID Score')
        plt.legend()
        plt.grid(True)

    # Time series comparison
    plt.subplot(4, 4, 11)
    real_sample = real_data[0].squeeze().cpu().numpy()
    synthetic_sample = synthetic_data[0].squeeze().cpu().numpy()
    plt.plot(real_sample, label='Real', alpha=0.8, linewidth=2)
    plt.plot(synthetic_sample, label='Synthetic (BUG-FIXED)', alpha=0.8, linewidth=2)
    plt.title('Sample Time Series Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # FFT comparison
    plt.subplot(4, 4, 12)
    real_fft = np.abs(np.fft.fft(real_sample))[:len(real_sample)//2]
    synthetic_fft = np.abs(np.fft.fft(synthetic_sample))[:len(synthetic_sample)//2]
    freqs = np.fft.fftfreq(len(real_sample))[:len(real_sample)//2]
    plt.semilogy(freqs, real_fft, label='Real FFT', alpha=0.8)
    plt.semilogy(freqs, synthetic_fft, label='Synthetic FFT (FIXED)', alpha=0.8)
    plt.title('Frequency Spectrum Comparison')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    # Statistical comparison
    plt.subplot(4, 4, 13)
    real_stats = [real_data.mean().item(), real_data.std().item()]
    synthetic_stats = [synthetic_data.mean().item(), synthetic_data.std().item()]
    x = ['Mean', 'Std']
    width = 0.35
    plt.bar([i - width/2 for i in range(len(x))], real_stats, width, label='Real', alpha=0.7)
    plt.bar([i + width/2 for i in range(len(x))], synthetic_stats, width, label='Synthetic', alpha=0.7)
    plt.title('Statistical Comparison (No Tanh)')
    plt.ylabel('Value')
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.grid(True)

    # DC bias final value
    plt.subplot(4, 4, 14)
    dc_value = generator.dc_bias.item()
    plt.bar(['DC Bias'], [dc_value], color='red', alpha=0.7)
    plt.title(f'Final DC Bias: {dc_value:.6f}')
    plt.ylabel('Value')
    plt.grid(True)

    # Loss component balance analysis
    plt.subplot(4, 4, 15)
    plt.plot(feat_losses, label='Feature Loss', alpha=0.7)
    plt.plot(stat_losses, label='Stat Loss', alpha=0.7)
    plt.title('Additional Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Loss ratios (to see balance)
    plt.subplot(4, 4, 16)
    if len(time_l1_losses) > 10:  # Only if we have enough data
        time_ratio = np.array(time_l1_losses) / np.array(mrstft_losses)
        plt.plot(time_ratio, label='Time-L1 / MR-STFT', alpha=0.7)
        plt.axhline(y=lambda_time/lambda_mrstft, color='red', linestyle='--', 
                   label=f'Target: {lambda_time/lambda_mrstft:.1f}')
        plt.title('Loss Component Balance')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('WGAN-GP/synthetic_data/analysis_v0.2.png', dpi=300, bbox_inches='tight')
    plt.show()

    return (generator, critic, {
        'critic_losses': critic_losses,
        'generator_losses': generator_losses,
        'mrstft_losses': mrstft_losses,
        'time_l1_losses': time_l1_losses,
        'deriv_l1_losses': deriv_l1_losses,
        'psd_losses': psd_losses,
        'acorr_losses': acorr_losses,
        'feat_losses': feat_losses,
        'stat_losses': stat_losses,
        'dc_reg_losses': dc_reg_losses,
        'fid_scores': fid_scores,
        'psd_metrics': psd_metrics,
        'best_fid_score': best_fid_score
    })

# Example usage with ALL CRITICAL BUG FIXES
if __name__ == "__main__":
    csv_file = 'data.csv'
    
    # Run main training with recommended settings
    results = train_fixed_wgan_gp(
        csv_path=csv_file,
        output_path='WGAN-GP/synthetic_data/data_model-v0.2.csv',
        epochs=200,
        batch_size=64,
        seq_len=128,
        noise_dim=64,
        lr_g=1e-4,      # Generator learning rate
        lr_c=2e-4,      # Critic learning rate
        lambda_gp=10,
        n_critic=5,
        # Recommended balanced settings
        lambda_adv=1.0,
        lambda_time=5.0,      # Reduced to prevent dominance
        lambda_deriv=0.5,
        lambda_mrstft=1.0,
        lambda_feat=1.0,
        lambda_stat=0.1,
        lambda_psd=0.2,
        lambda_acorr=0.2
    )
    