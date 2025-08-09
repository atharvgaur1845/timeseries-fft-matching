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
import os
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, sampling_freq=1.0):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.sampling_freq = sampling_freq
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        self.freq_embedding = nn.Parameter(torch.randn(d_model) * 0.1)
        self.temporal_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        if x.dim() == 3: 
            batch_size, seq_len, d_model = x.size()
        else:
            seq_len, batch_size, d_model = x.size()
        pos_enc = self.pos_embedding[:seq_len]
        time_steps = torch.arange(seq_len, device=x.device, dtype=torch.float)
        freq_component = torch.outer(time_steps * self.sampling_freq, self.freq_embedding)
        freq_component = torch.sin(freq_component * self.temporal_scale)
        
        combined_encoding = pos_enc + freq_component
        
        if x.dim() == 3:
            combined_encoding = combined_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            combined_encoding = combined_encoding.unsqueeze(1).expand(-1, batch_size, -1)
            
        return x + combined_encoding

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             dilation=dilation, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class TransformerGenerator(nn.Module):
    def __init__(self, noise_dim=100, seq_len=64, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, sampling_freq=1.0):
        super(TransformerGenerator, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.noise_dim = noise_dim
        self.sampling_freq = sampling_freq
        self.input_projection = nn.Sequential(
            nn.Linear(noise_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.pos_encoding = LearnablePositionalEncoding(d_model, seq_len, sampling_freq)
        self.temporal_convs = nn.ModuleList([
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=1),
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=2),
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=4)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()
        )
        
        self.freq_regularizer = nn.Parameter(torch.ones(seq_len // 2 + 1))
        
    def forward(self, noise):
        batch_size = noise.size(0)
        x = self.input_projection(noise)
        x = self.pos_encoding(x)
        
        conv_outputs = []
        conv_x = x
        for conv_layer in self.temporal_convs:
            conv_x = conv_layer(conv_x)
            conv_outputs.append(conv_x)
    
        x = x + sum(conv_outputs) / len(conv_outputs)
    
        x = self.transformer(x)
        output = self.output_projection(x)
        
        return output

class TransformerCritic(nn.Module):
    def __init__(self, seq_len=64, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, sampling_freq=1.0):
        super(TransformerCritic, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.sampling_freq = sampling_freq
        
        self.input_projection = nn.utils.spectral_norm(
            nn.Linear(1, d_model)
        )
        
        self.pos_encoding = LearnablePositionalEncoding(d_model, seq_len, sampling_freq)
        
        self.temporal_convs = nn.ModuleList([
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=1),
            TemporalConvBlock(d_model, d_model, kernel_size=5, dilation=2),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=4)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(d_model * 2, d_model)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(d_model, d_model // 2)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(d_model // 2, 1))
        )
        
    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_projection(x)
        
        x = self.pos_encoding(x)
        
        conv_outputs = []
        conv_x = x
        for conv_layer in self.temporal_convs:
            conv_x = conv_layer(conv_x)
            conv_outputs.append(conv_x)
        
        x = x + sum(conv_outputs) / len(conv_outputs)
        x = self.transformer(x)
        
        global_feat = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        attn_out, _ = self.attention_pool(x, x, x)
        attn_feat = torch.mean(attn_out, dim=1)
        
        combined_feat = torch.cat([global_feat, attn_feat], dim=1)
        
        output = self.classifier(combined_feat)
        
        return output

class SpectralFeatureExtractor(nn.Module):
    def __init__(self, seq_len=64, sampling_freq=1.0, n_features=512):
        super(SpectralFeatureExtractor, self).__init__()
        
        self.seq_len = seq_len
        self.sampling_freq = sampling_freq
        self.n_features = n_features
        
        self.n_fft = seq_len
        self.hop_length = seq_len // 4
        
        self.feature_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, n_features)
        )
        
    def extract_spectral_features(self, x):
        x_np = x.squeeze(-1).cpu().numpy()
        
        spectral_features = []
        for sample in x_np:
            freqs, psd = signal.welch(sample, fs=self.sampling_freq, nperseg=min(len(sample), 64))
            
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
            spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
            
            features = np.concatenate([
                psd[:32],
                [spectral_centroid, spectral_bandwidth, spectral_rolloff]
            ])
            spectral_features.append(features)
        
        return torch.FloatTensor(spectral_features).to(x.device)
    
    def forward(self, x):
        x_conv = x.transpose(1, 2)  
        temporal_features = self.feature_net(x_conv)
        
        spectral_features = self.extract_spectral_features(x)
        
        combined = torch.cat([temporal_features, spectral_features], dim=1)
        
        return combined

def frequency_domain_loss(real_data, fake_data, sampling_freq=1.0):
    real_fft = torch.fft.fft(real_data.squeeze(-1), dim=1)
    fake_fft = torch.fft.fft(fake_data.squeeze(-1), dim=1)
    real_psd = torch.abs(real_fft) ** 2
    fake_psd = torch.abs(fake_fft) ** 2
    
    freq_loss = F.l1_loss(fake_psd, real_psd)
    real_phase = torch.angle(real_fft)
    fake_phase = torch.angle(fake_fft)
    phase_loss = F.l1_loss(torch.sin(fake_phase), torch.sin(real_phase))
    
    return freq_loss + 0.1 * phase_loss

def calculate_proper_fid(real_features, generated_features):
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().detach().numpy()
    if torch.is_tensor(generated_features):
        generated_features = generated_features.cpu().detach().numpy()
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    eps = 1e-6
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def gradient_penalty(critic, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)

    alpha = torch.rand(batch_size, 1, 1).to(device)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
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

def load_time_series_data_with_freq(csv_path, seq_len=64, sampling_freq=1.0):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 0].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data)
    stride = seq_len // 4  
    sequences = []
    for i in range(0, len(normalized_data) - seq_len + 1, stride):
        sequences.append(normalized_data[i:i + seq_len])
    
    sequences = np.array(sequences)
    
    return torch.FloatTensor(sequences), scaler, sampling_freq

def train_wgan_gp(csv_path, output_path='enhanced_synthetic_data.csv', 
                          epochs=2000, batch_size=64, seq_len=128, noise_dim=128, 
                          lr=2e-4, lambda_gp=10, lambda_freq=1.0, n_critic=3, 
                          sampling_freq=1.0, fid_eval_freq=100):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Sampling frequency: {sampling_freq} Hz")
    
    real_data, scaler, sampling_freq = load_time_series_data_with_freq(csv_path, seq_len, sampling_freq)
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    feature_extractor = SpectralFeatureExtractor(seq_len, sampling_freq).to(device)
    generator = TransformerGenerator(
        noise_dim=noise_dim,
        seq_len=seq_len,
        d_model=256,
        nhead=8,
        num_layers=6,
        sampling_freq=sampling_freq
    ).to(device)
    
    critic = TransformerCritic(
        seq_len=seq_len,
        d_model=256,
        nhead=8,
        num_layers=6,
        sampling_freq=sampling_freq
    ).to(device)
    
    optimizer_g = optim.AdamW(generator.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-4)
    optimizer_c = optim.AdamW(critic.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-4)
    
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=epochs)
    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=epochs)
    
    critic_losses = []
    generator_losses = []
    freq_losses = []
    fid_scores = []
    fid_epochs = []
    
    best_fid_score = float('inf')
    best_model_path = 'best_enhanced_wgan_gp.pth'
    
    
    for epoch in range(epochs):
        epoch_critic_loss = 0
        epoch_generator_loss = 0
        epoch_freq_loss = 0
        num_batches = 0
        
        for batch_idx, (real_samples,) in enumerate(dataloader):
            real_samples = real_samples.to(device)
            current_batch_size = real_samples.size(0)
        
            for _ in range(n_critic):
                optimizer_c.zero_grad()
                
                noise = torch.randn(current_batch_size, seq_len, noise_dim).to(device)
                fake_samples = generator(noise).detach()
                
                real_output = critic(real_samples)
                fake_output = critic(fake_samples)
                
                gp = gradient_penalty(critic, real_samples, fake_samples, device)
                
                critic_loss = fake_output.mean() - real_output.mean() + lambda_gp * gp
                critic_loss.backward()
    
                
                optimizer_c.step()
                epoch_critic_loss += critic_loss.item()

            optimizer_g.zero_grad()
            
            noise = torch.randn(current_batch_size, seq_len, noise_dim).to(device)
            fake_samples = generator(noise)
            fake_output = critic(fake_samples)
        
            wasserstein_loss = -fake_output.mean()
        
            freq_loss = frequency_domain_loss(real_samples, fake_samples, sampling_freq)
            
            generator_loss = wasserstein_loss + lambda_freq * freq_loss
            generator_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_g.step()
            
            epoch_generator_loss += wasserstein_loss.item()
            epoch_freq_loss += freq_loss.item()
            num_batches += 1
        
        scheduler_g.step()
        scheduler_c.step()
        
        avg_critic_loss = epoch_critic_loss / (num_batches * n_critic)
        avg_generator_loss = epoch_generator_loss / num_batches
        avg_freq_loss = epoch_freq_loss / num_batches
        
        critic_losses.append(avg_critic_loss)
        generator_losses.append(avg_generator_loss)
        freq_losses.append(avg_freq_loss)
        
        if epoch % fid_eval_freq == 0 and epoch > 0:
            print(f"Evaluating at epoch {epoch}...")
            generator.eval()
            feature_extractor.eval()
            
            real_features_list = []
            fake_features_list = []
            
            with torch.no_grad():
                for batch_data, in dataloader:
                    batch_data = batch_data.to(device)
                    current_batch_size = batch_data.size(0)
                    
                    # Real features
                    real_feat = feature_extractor(batch_data)
                    real_features_list.append(real_feat)
                    
                    # Fake features
                    noise = torch.randn(current_batch_size, seq_len, noise_dim).to(device)
                    fake_data = generator(noise)
                    fake_feat = feature_extractor(fake_data)
                    fake_features_list.append(fake_feat)
            
            real_features = torch.cat(real_features_list, dim=0)
            fake_features = torch.cat(fake_features_list, dim=0)
            
            fid_score = calculate_proper_fid(real_features, fake_features)
            fid_scores.append(fid_score)
            fid_epochs.append(epoch)
            
            if fid_score < best_fid_score:
                best_fid_score = fid_score
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'fid_score': fid_score,
                    'sampling_freq': sampling_freq
                }, best_model_path)
                print(f"✓ New best model saved! FID: {fid_score:.4f}")
            
            generator.train()
            feature_extractor.train()
        
        # Print progress
        if epoch % 50 == 0:
            current_lr = scheduler_g.get_last_lr()[0]
            print(f"Epoch [{epoch}/{epochs}] - Critic: {avg_critic_loss:.4f}, "
                  f"Generator: {avg_generator_loss:.4f}, Freq: {avg_freq_loss:.4f}, "
                  f"LR: {current_lr:.6f}")
    
    print(f"Training completed! Best FID: {best_fid_score:.4f}")
    
    # Load best model and generate final data
    checkpoint = torch.load(best_model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    generator.eval()
    with torch.no_grad():
        num_samples = len(real_data)
        all_synthetic = []
        
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch, seq_len, noise_dim).to(device)
            synthetic_batch = generator(noise)
            all_synthetic.append(synthetic_batch)
        
        synthetic_data = torch.cat(all_synthetic, dim=0)

    synthetic_flat = synthetic_data.cpu().numpy().reshape(-1, 1)
    denormalized = scaler.inverse_transform(synthetic_flat)
    
    df = pd.DataFrame(denormalized, columns=['enhanced_synthetic_timeseries'])
    df.to_csv(output_path, index=False)
    print(f"Enhanced synthetic data saved to {output_path}")
    
    plt.figure(figsize=(20, 12))
    
    plt.subplot(3, 3, 1)
    plt.plot(critic_losses, label='Critic Loss', alpha=0.8)
    plt.title('Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 3, 2)
    plt.plot(generator_losses, label='Generator Loss', alpha=0.8, color='orange')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 3, 3)
    plt.plot(freq_losses, label='Frequency Loss', alpha=0.8, color='green')
    plt.title('Frequency Domain Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 3, 4)
    if fid_scores:
        plt.plot(fid_epochs, fid_scores, 'bo-', alpha=0.8)
        plt.axhline(y=best_fid_score, color='red', linestyle='--', 
                   label=f'Best: {best_fid_score:.4f}')
    plt.title('FID Score Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 3, 5)
    real_sample = real_data[0].squeeze().cpu().numpy()
    synthetic_sample = synthetic_data[0].squeeze().cpu().numpy()
    plt.plot(real_sample[:200], label='Real', alpha=0.8)
    plt.plot(synthetic_sample[:200], label='Synthetic', alpha=0.8)
    plt.title('Sample Time Series Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 6)
    real_fft = np.abs(np.fft.fft(real_sample))[:len(real_sample)//2]
    synthetic_fft = np.abs(np.fft.fft(synthetic_sample))[:len(synthetic_sample)//2]
    freqs = np.fft.fftfreq(len(real_sample), 1/sampling_freq)[:len(real_sample)//2]
    plt.semilogy(freqs, real_fft, label='Real FFT', alpha=0.8)
    plt.semilogy(freqs, synthetic_fft, label='Synthetic FFT', alpha=0.8)
    plt.title('Frequency Domain Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 7)
    plt.hist(real_data.flatten().cpu().numpy(), bins=50, alpha=0.7, 
             label='Real', density=True)
    plt.hist(synthetic_data.flatten().cpu().numpy(), bins=50, alpha=0.7, 
             label='Synthetic', density=True)
    plt.title('Value Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 8)
    from scipy.signal import correlate
    real_autocorr = correlate(real_sample, real_sample, mode='full')
    synthetic_autocorr = correlate(synthetic_sample, synthetic_sample, mode='full')
    lags = np.arange(-len(real_sample)+1, len(real_sample))
    center = len(lags) // 2
    plt.plot(lags[center:center+100], real_autocorr[center:center+100], 
             label='Real', alpha=0.8)
    plt.plot(lags[center:center+100], synthetic_autocorr[center:center+100], 
             label='Synthetic', alpha=0.8)
    plt.title('Autocorrelation Comparison')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 9)
    lrs = [scheduler_g.get_last_lr()[0] * (0.5 ** (i // (epochs // 10))) for i in range(epochs)]
    plt.plot(lrs, alpha=0.8)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('enhanced_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return (generator, critic, critic_losses, generator_losses, 
            freq_losses, fid_scores, best_fid_score)

if __name__ == "__main__":
    
    results = train_wgan_gp(
        csv_path='data.csv',
        output_path='enhanced_synthetic_timeseries.csv',
        epochs=1000,
        batch_size=64,
        seq_len=128,
        noise_dim=128,
        lr=2e-4,
        lambda_gp=10,
        lambda_freq=2.0, 
        n_critic=3,
        sampling_freq=25600.0,
        fid_eval_freq=10
    )
    
    