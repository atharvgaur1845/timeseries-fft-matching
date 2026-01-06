import pandas as pd
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import pearsonr, entropy, ttest_ind, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import csv
import torch.nn as nn
import os
import random
from torch.utils.data import Dataset, ConcatDataset
import torch.optim as optim
import torch.autograd as autograd
import time
import math
from torch.nn.utils import spectral_norm

torch.backends.cudnn.benchmark = True

path = 'CWRU_data'
num_classes = 12
percentage = 15  # More training data
num_train_samples = (4800 * percentage) // 100
num_blocks = 0
fs = 12000
csv_files = glob.glob(f'{path}*.csv')

class SingleCSVDataset(Dataset):
    def __init__(self, file_path, seq_len=1824, num_train_samples=1000, num_test_samples=400):
        filename = os.path.splitext(os.path.basename(file_path))[0].replace("_Sensor1", "")
        label_mapping = {
            "N": 0, "7BA": 1, "7IR": 2, "7OR": 3, "14BA": 4, "14IR": 5, 
            "14OR": 6, "21BA": 7, "21IR": 8, "21OR": 9, "BA28": 10, "IR28": 11
        }
        
        label = label_mapping.get(filename, -1)
        if label == -1:
            raise ValueError(f"Label not found for filename: {filename}")
        
        df = pd.read_csv(file_path, header=None)
        values = df.to_numpy().flatten().astype(np.float32)
        
        self.min_val = np.min(values)
        self.max_val = np.max(values)
        values = 2 * (values - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1
        
        self.data = torch.tensor(values, dtype=torch.float32)
        self.seq_len = seq_len
        
        sequences = self._create_sequences(self.data, seq_len, stride=seq_len // 50)
        print(f"Number of sequences in {file_path}: {len(sequences)}")
        
        random.seed(42)
        random.shuffle(sequences)
        
        if len(sequences) < num_train_samples + num_test_samples:
            raise ValueError(f"Not enough samples in {file_path}. Required: {num_train_samples + num_test_samples}, Found: {len(sequences)}")
        
        self.train_sequences = sequences[:num_train_samples]
        self.test_sequences = sequences[num_train_samples:num_train_samples + num_test_samples]
        self.valid_sequences = sequences[num_train_samples + num_test_samples:num_train_samples + (2 * num_test_samples)]
        self.label = torch.tensor(label, dtype=torch.long)
    
    def _create_sequences(self, data, seq_len, stride=None):
        if stride is None:
            stride = seq_len
        sequences = [data[i : i + seq_len] for i in range(0, len(data) - seq_len, stride)]
        return sequences
    
    def __len__(self):
        return len(self.train_sequences) + len(self.test_sequences)
    
    def __getitem__(self, idx):
        if idx < len(self.train_sequences):
            sequence = self.train_sequences[idx].unsqueeze(0)
        elif idx < len(self.train_sequences) + len(self.test_sequences) and idx >= len(self.train_sequences):
            sequence = self.test_sequences[idx - len(self.train_sequences)].unsqueeze(0)
        else:
            sequence = self.valid_sequences[idx - len(self.train_sequences) - len(self.test_sequences)].unsqueeze(0)
        
        return sequence, self.label

def load_all_data(csv_folder, seq_len=1824, num_train_samples=100, num_test_samples=200):
    csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
    train_datasets, test_datasets, valid_datasets = [], [], []
    
    for file_path in csv_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        train_samples = num_train_samples if filename[0] != 'N' else 4800
        test_samples = num_test_samples if filename[0] != 'N' else 200
        
        dataset = SingleCSVDataset(file_path, seq_len, train_samples, test_samples)
        train_datasets.append(torch.utils.data.Subset(dataset, range(train_samples)))
        test_datasets.append(torch.utils.data.Subset(dataset, range(train_samples, train_samples + test_samples)))
        valid_datasets.append(torch.utils.data.Subset(dataset, range(train_samples + test_samples, train_samples + 2 * test_samples)))
    
    return ConcatDataset(train_datasets), ConcatDataset(test_datasets), ConcatDataset(valid_datasets)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=8, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.num_heads = num_heads
        
        # ENHANCEMENT 1: Multi-head attention with spectral normalization
        self.query = spectral_norm(nn.Conv1d(in_channels, in_channels // reduction, 1))
        self.key = spectral_norm(nn.Conv1d(in_channels, in_channels // reduction, 1))
        self.value = spectral_norm(nn.Conv1d(in_channels, in_channels, 1))
        
        # ENHANCEMENT 2: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(in_channels)
        
        # ENHANCEMENT 3: Temperature scaling for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
        # ENHANCEMENT 4: Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, in_channels, 1824) * 0.02)
        
        # ENHANCEMENT 5: Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # ENHANCEMENT 6: Residual scaling
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # Additional scaling
        
    def forward(self, x):
        batch_size, channels, length = x.size()
        
        # Add positional encoding
        if length <= self.pos_embedding.size(2):
            x = x + self.pos_embedding[:, :, :length] * self.alpha
        
        # Layer normalization first (pre-norm architecture)
        x_norm = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Generate query, key, value
        Q = self.query(x_norm).view(batch_size, -1, length).permute(0, 2, 1)  # (B, L, C//r)
        K = self.key(x_norm).view(batch_size, -1, length)  # (B, C//r, L)
        V = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)  # (B, L, C)
        
        # Scaled dot-product attention with temperature
        attention = torch.bmm(Q, K) / (math.sqrt(self.in_channels // self.reduction) * self.temperature)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention
        out = torch.bmm(attention, V)  # (B, L, C)
        out = out.permute(0, 2, 1).view(batch_size, channels, length)
        
        # Enhanced residual connection
        return self.gamma * out + x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        
        # ENHANCEMENT 1: Pre-normalization architecture
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation))
        
        # ENHANCEMENT 2: GELU activation (better than ReLU)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(out_channels)
        self.conv2 = spectral_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation))
        
        # ENHANCEMENT 3: Squeeze-and-Excitation attention
        self.se = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),  # <-- ADD THIS to convert (B, C, 1) -> (B, C)
        nn.Linear(out_channels, out_channels // 16),
        nn.GELU(),
        nn.Linear(out_channels // 16, out_channels),
        nn.Sigmoid()
        )
        
        # ENHANCEMENT 4: Residual projection with proper initialization
        self.residual = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1)) if in_channels != out_channels else nn.Identity()
        
        # ENHANCEMENT 5: Layer scaling for better training
        self.layer_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        res = self.residual(x)
        
        # Pre-normalization
        x_norm1 = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv1(x_norm1)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x_norm2 = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv2(x_norm2)
        
        # Squeeze-and-excitation
        b, c, l = x.shape
        se_weights = self.se(x).unsqueeze(-1)
        x = x * se_weights
        
        x = self.gelu(x)
        x = self.dropout(x)
        
        # Scaled residual connection
        return res + self.layer_scale * x

class AttentionTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        self.tcn = TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
        self.attention = SelfAttention(out_channels)
        
        # ENHANCEMENT: Feature fusion with gating
        self.fusion_gate = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),  # <-- ADD THIS
        nn.Linear(out_channels, out_channels // 4),
        nn.GELU(),
        nn.Linear(out_channels // 4, 1),
        nn.Sigmoid()
        )
        
        # Layer normalization for final output
        self.final_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        # TCN processing
        tcn_out = self.tcn(x)
        
        # Self-attention processing
        attn_out = self.attention(tcn_out)
        
        # Adaptive fusion with gating
        b, c, l = attn_out.shape
        gate = self.fusion_gate(attn_out).unsqueeze(-1)
        
        # Gated combination
        fused = gate * attn_out + (1 - gate) * tcn_out
        
        # Final normalization
        fused_norm = self.final_norm(fused.transpose(1, 2)).transpose(1, 2)
        
        return fused_norm

class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=12, embed_size=16, num_blocks=2,
                 channels=128, kernel_size=5, dropout=0.05, output_length=1824):
        super().__init__()
        
        self.output_length = output_length
        self.num_classes = num_classes
        self.channels = channels
        
        # ENHANCEMENT 1: Improved label embedding with class-aware features
        self.label_emb = nn.Embedding(num_classes, embed_size)
        self.label_proj = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.GELU(),
            nn.LayerNorm(embed_size * 2),
            nn.Linear(embed_size * 2, embed_size)
        )
        
        # ENHANCEMENT 2: Better noise processing
        self.noise_proj = nn.Sequential(
            spectral_norm(nn.Linear(nz, nz * 2)),
            nn.GELU(),
            nn.LayerNorm(nz * 2),
            spectral_norm(nn.Linear(nz * 2, nz))
        )
        
        # Keep successful upsampling approach
        self.initial_length = 114
        self.init_conv = spectral_norm(nn.ConvTranspose1d(nz + embed_size, channels, kernel_size=self.initial_length, stride=1, padding=0))
        self.init_bn = nn.BatchNorm1d(channels)
        self.init_gelu = nn.GELU()  # Better activation
        
        # Enhanced upsampling with better normalization
        self.upsample_layers = nn.Sequential(
            # 114 -> 228
            spectral_norm(nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            # 228 -> 456
            spectral_norm(nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            # 456 -> 912
            spectral_norm(nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            # 912 -> 1824
            spectral_norm(nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        
        # ENHANCEMENT 3: Ultimate TCN blocks with superior attention
        self.tcn_blocks = nn.Sequential(*[
            AttentionTCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        
        # ENHANCEMENT 4: Global context with temporal awareness
        self.global_attention = SelfAttention(channels)
        
        # ENHANCEMENT 5: Temporal consistency loss preparation
        self.temporal_proj = spectral_norm(nn.Conv1d(channels, channels//2, kernel_size=3, padding=1))
        
        # Keep successful downsampling approach with enhancements
        self.downsample_layers = nn.Sequential(
            spectral_norm(nn.Conv1d(channels, channels//2, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm1d(channels//2),
            nn.GELU(),
            spectral_norm(nn.Conv1d(channels//2, channels//4, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm1d(channels//4),
            nn.GELU(),
            spectral_norm(nn.Conv1d(channels//4, channels//8, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm1d(channels//8),
            nn.GELU(),
            spectral_norm(nn.Conv1d(channels//8, 1, kernel_size=1, stride=1, padding=0)),
        )
        
        self.tanh = nn.Tanh()
        
        # ENHANCEMENT 6: Learnable output scaling with class conditioning
        self.output_scale = nn.Parameter(torch.ones(1))
        self.class_scale = nn.Embedding(num_classes, 1)
        
        # Apply better weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
    
    def forward(self, z, labels):
        # Enhanced label processing
        label_embedding = self.label_emb(labels)
        label_embedding = self.label_proj(label_embedding)
        
        # Enhanced noise processing
        z_processed = self.noise_proj(z)
        
        # Combine and create initial sequence
        x = torch.cat([z_processed, label_embedding], dim=1).unsqueeze(2)
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_gelu(x)
        
        # Upsampling to target length
        x = self.upsample_layers(x)
        
        # Ensure target length
        if x.size(2) != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
        
        # Ultimate TCN processing with attention
        x = self.tcn_blocks(x)
        
        # Global attention
        x = self.global_attention(x)
        
        # Final downsampling
        x = self.downsample_layers(x)
        
        # Ensure final length
        if x.size(2) != self.output_length:
            x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
        
        # Class-conditional output scaling
        class_scaling = self.class_scale(labels).unsqueeze(-1)
        
        return self.tanh(x * self.output_scale * class_scaling)

class DiscriminatorTCN(nn.Module):
    def __init__(self, num_classes=12, num_blocks=9, channels=128, kernel_size=5, dropout=0.05):
        super().__init__()
        
        # Enhanced input processing
        self.initial_conv = spectral_norm(nn.Conv1d(1, channels, kernel_size=7, padding=3))
        self.initial_norm = nn.LayerNorm(channels)
        
        # Superior TCN layers
        self.tcn_layers = nn.Sequential(*[
            TCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout) 
            for i in range(num_blocks)
        ])
        
        # Enhanced self-attention
        self.attention = SelfAttention(channels)
        
        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            spectral_norm(nn.Conv1d(channels, channels//2, kernel_size=5, padding=2)),
            nn.LayerNorm(channels//2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Output heads with better architecture
        feature_dim = channels//2
        self.adv_output = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, feature_dim//2)),
            nn.GELU(),
            spectral_norm(nn.Linear(feature_dim//2, 1))
        )
        
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, feature_dim//2)),
            nn.GELU(),
            spectral_norm(nn.Linear(feature_dim//2, num_classes))
        )
        
    def forward(self, x):
        # Enhanced input processing
        x = self.initial_conv(x)
        x = self.initial_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # TCN processing
        x = self.tcn_layers(x)
        
        # Self-attention
        x = self.attention(x)
        
        # Temporal feature extraction
        x = self.temporal_conv(x).squeeze(2)
        
        # Output predictions
        validity = self.adv_output(x)
        label_pred = self.classifier(x)
        
        return validity, label_pred

def weights_init(m):
    """Enhanced weight initialization"""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
    elif 'BatchNorm' in classname or 'LayerNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif 'Linear' in classname:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def compute_gradient_penalty(netD, real_data, fake_data):
    """Enhanced gradient penalty with better stability"""
    device = real_data.device
    b_size = real_data.size(0)
    eps = torch.rand(b_size, 1, 1, device=device)
    interpolates = eps * real_data + (1 - eps) * fake_data
    interpolates.requires_grad_(True)
    
    d_interpolates, _ = netD(interpolates)
    d_interpolates = d_interpolates.view(-1)
    
    grad_outputs = torch.ones(d_interpolates.shape, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(b_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = 15 * ((grad_norm - 1) ** 2).mean()  # Enhanced penalty coefficient
    
    return gp

def temporal_consistency_loss(generated_data):
    """Additional loss for temporal consistency"""
    # Compute temporal differences
    diff = generated_data[:, :, 1:] - generated_data[:, :, :-1]
    # Encourage smooth temporal transitions
    return torch.mean(torch.abs(diff))

def spectral_loss(original_fft, generated_fft):
    """Spectral domain loss for better frequency matching"""
    return F.mse_loss(generated_fft, original_fft)

def train_ultimate_model(dataloader, save_path=""):
    """Ultimate training with multiple enhancements"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    lr_g, lr_d = 8e-5, 1.5e-4  # Slightly adjusted learning rates
    epoch_num = 50  
    n_critic = 4   # Balanced training
    p_coeff = 15   # Enhanced gradient penalty
    cls_coeff = 2  # Stronger class conditioning
    temp_coeff = 0.1  # Temporal consistency weight
    
    netD = DiscriminatorTCN(num_classes=num_classes, num_blocks=num_blocks, channels=128).to(device)
    netG = Generator(100, num_classes=num_classes, channels=128).to(device)
    
    # Enhanced weight initialization
    netD.apply(weights_init)
    netG.apply(weights_init)
    
    # Enhanced optimizers with better parameters
    optimizerD = optim.AdamW(netD.parameters(), lr=lr_d, betas=(0.0, 0.95), weight_decay=0.01)
    optimizerG = optim.AdamW(netG.parameters(), lr=lr_g, betas=(0.0, 0.95), weight_decay=0.01)
    
    # Learning rate schedulers for better convergence
    schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=epoch_num, eta_min=1e-6)
    schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=epoch_num, eta_min=1e-6)
    
    # Enhanced loss functions
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    for epoch in range(epoch_num):
        for step, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)
            b_size = real_data.size(0)
            
            # Train Discriminator with enhanced approach
            netD.zero_grad()
            
            # Generate fake data conditioned on labels
            noise = torch.randn(b_size, 100, device=device)
            fake_data = netG(noise, labels).detach()
            
            # Get Discriminator predictions
            d_real, class_logits_real = netD(real_data)
            d_fake, _ = netD(fake_data)
            
            # Enhanced Wasserstein loss
            loss_D_wasserstein = torch.mean(d_fake) - torch.mean(d_real)
            
            # Classification loss with label smoothing
            loss_D_class = class_criterion(class_logits_real, labels) * cls_coeff
            
            # Enhanced gradient penalty
            gp = compute_gradient_penalty(netD, real_data, fake_data)
            
            # Total Discriminator loss
            loss_D = loss_D_wasserstein + gp + loss_D_class
            loss_D.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()
            
            # Train Generator with enhanced losses
            if step % n_critic == 0:
                netG.zero_grad()
                
                # Generate fake samples
                noise = torch.randn(b_size, 100, device=device)
                fake_data = netG(noise, labels)
                
                # Discriminator output for fake samples
                d_fake, class_logits_fake = netD(fake_data)
                
                # Enhanced Generator losses
                loss_G_wasserstein = -torch.mean(d_fake)
                loss_G_class = class_criterion(class_logits_fake, labels) * cls_coeff
                
                # Temporal consistency loss
                loss_G_temporal = temporal_consistency_loss(fake_data) * temp_coeff
                
                # Total Generator loss
                loss_G = loss_G_wasserstein + loss_G_class + loss_G_temporal
                loss_G.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                optimizerG.step()
            
            if step % 5 == 0:
                print(f"[Epoch {epoch}/{epoch_num}][Step {step}/{len(dataloader)}] "
                      f"Loss_D: {loss_D.item():.4f} (W: {loss_D_wasserstein.item():.4f}, "
                      f"GP: {gp.item():.4f}, Cls: {loss_D_class.item():.4f}) | "
                      f"Loss_G: {loss_G.item():.4f} (W: {loss_G_wasserstein.item():.4f}, "
                      f"Cls: {loss_G_class.item():.4f}, Temp: {loss_G_temporal.item():.4f})")
        
        # Update learning rates
        schedulerD.step()
        schedulerG.step()
        
        # Save models
        torch.save(netG.state_dict(), f"{save_path}generator_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"{save_path}discriminator_epoch_{epoch}.pth")
        print(f"Epoch {epoch} | Models saved")
    
    return netG, netD

# Keep all evaluation functions from successful model
def load_data(csv_path):
    values = pd.read_csv(csv_path).values.flatten().astype(np.float32)
    values = 2 * (values - values.min()) / (values.max() - values.min() + 1e-8) - 1
    return values

def compute_fft(signal):
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft_vals), d=1/fs)
    pos_idx = freqs > 0
    return np.abs(fft_vals[pos_idx]), freqs[pos_idx]

def load_generator(GeneratorModel, num_classes, weight_path, nz=100, device='cpu'):
    netG = GeneratorModel(nz, num_classes=num_classes).to(device)
    netG.load_state_dict(torch.load(weight_path, map_location=device))
    netG.eval()
    return netG

def generate_sequences(generator, label, n_samples=1, nz=100, device='cpu'):
    noise = torch.randn(n_samples, nz, device=device)
    label_tensor = torch.full((n_samples,), label, dtype=torch.long, device=device)
    with torch.no_grad():
        return generator(noise, label_tensor).cpu().numpy().squeeze()

def compute_mmd(x, y, sigma=1.0):
    x, y = torch.tensor(x).unsqueeze(1), torch.tensor(y).unsqueeze(1)
    kernel = lambda a, b: torch.exp(-((a - b.T) ** 2) / (2 * sigma ** 2))
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()

def compute_metrics(original, generated):
    pearson_corr, _ = pearsonr(original, generated)
    cosine_sim = cosine_similarity(original.reshape(1, -1), generated.reshape(1, -1))[0][0]
    kl_div = entropy(original / original.sum(), generated / generated.sum())
    mmd_val = compute_mmd(original, generated).item()
    t_stat, p_value = ttest_ind(original, generated)
    return pearson_corr, cosine_sim, kl_div, mmd_val, t_stat, p_value

def plot_ffts_all(original_freqs, original_mags, generated_freqs, generated_mags, titles):
    plt.figure(figsize=(12, 16))
    for i in range(min(10, len(titles))):
        plt.subplot(5, 2, i + 1)
        plt.plot(original_freqs[i], original_mags[i], label="Original FFT", color='black')
        plt.plot(generated_freqs[i], generated_mags[i], label="Generated FFT", color='red', linestyle='--')
        plt.title(titles[i].replace("_Sensor", "").replace("30hz", ""))
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0, fs // 8)
        plt.ylabel("Magnitude")
        plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    csv_folder = f"{path}"
    train_data, test_data, valid_data = load_all_data(csv_folder, num_train_samples=num_train_samples, num_test_samples=200)
    print(f"Total training samples: {len(train_data)}")
    print(f"Total testing samples: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)
    
    # Training
    print("Starting model training...")
    start_time = time.time()
    
    trained_generator, trained_discriminator = train_ultimate_model(train_loader)
    
    training_time = time.time() - start_time
    np.save(f"{num_train_samples}training_time.npy", training_time)
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluation (same as successful model)
    print("Starting evaluation...")
    label_mapping = {
        "N": 0, "7BA": 1, "7IR": 2, "7OR": 3, "14BA": 4, "14IR": 5, 
        "14OR": 6, "21BA": 7, "21IR": 8, "21OR": 9, "BA28": 10, "IR28": 11
    }
    
    weight_path = f'generator_epoch_59.pth'
    results_csv_path = f"{num_train_samples}_generation_results.csv"
    
    original_freqs_all, original_mags_all = [], []
    generated_freqs_all, generated_mags_all = [], []
    
    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Avg Pearson", "Avg Cosine", "Avg KL", "Avg MMD", "Avg GAN-Test P-value"])
        
        for file_name, label_value in label_mapping.items():
            print(f"Evaluating {file_name}...")
            pearson_corrs, cosine_sims, kl_divs, mmd_vals, p_values = [], [], [], [], []
            generated_samples = []
            
            generator = load_generator(Generator, num_classes, weight_path, device=device)
            
            for _ in range(100):
                indexes = [idx for idx, (_, label) in enumerate(test_data) if label == label_value]
                random_index = np.random.choice(indexes)
                original_signal = test_data[random_index][0].numpy().flatten()
                original_mag, original_freq = compute_fft(original_signal)
                
                generated_signal = generate_sequences(generator, label_value, n_samples=1, device=device)
                if generated_signal.ndim > 1:
                    generated_signal = generated_signal[0]
                
                generated_samples.append(generated_signal)
                generated_mag, generated_freq = compute_fft(generated_signal)
                
                pearson_corr, cosine_sim, kl_div, mmd_val, _, p_value = compute_metrics(original_mag, generated_mag)
                
                pearson_corrs.append(pearson_corr)
                cosine_sims.append(cosine_sim)
                kl_divs.append(kl_div)
                mmd_vals.append(mmd_val)
                p_values.append(p_value)
            
            # Calculate averages
            avg_pearson = np.mean(pearson_corrs)
            avg_cosine = np.mean(cosine_sims)
            avg_kl = np.mean(kl_divs)
            avg_mmd = np.mean(mmd_vals)
            avg_p_value = np.mean(p_values)
            
            writer.writerow([file_name, avg_pearson, avg_cosine, avg_kl, avg_mmd, avg_p_value])
            print(f"{file_name}: Pearson={avg_pearson:.4f}, Cosine={avg_cosine:.4f}, "
                  f"KL={avg_kl:.4f}, MMD={avg_mmd:.4f}, P-value={avg_p_value:.4f}")
            
            # Store for plotting
            original_freqs_all.append(original_freq)
            original_mags_all.append(original_mag)
            generated_freqs_all.append(generated_freq)
            generated_mags_all.append(generated_mag)
            
            # Save generated samples
            generated_samples_arr = np.stack(generated_samples, axis=1)
            gen_csv_path = f"{num_train_samples}_ultimate_generated_{file_name}.csv"
            np.savetxt(gen_csv_path, generated_samples_arr, delimiter=",")
            print(f"Generated samples for {file_name} saved to {gen_csv_path}")
    
    # Plot results
    plot_ffts_all(original_freqs_all, original_mags_all, 
                  generated_freqs_all, generated_mags_all, 
                  list(label_mapping.keys()))
    
    print(f"results stored in {results_csv_path}")
