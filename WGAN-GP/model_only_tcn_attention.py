import pandas as pd
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.optim as optim
import torch.autograd as autograd
import time
import math
from scipy.stats import pearsonr, entropy, ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils import spectral_norm

torch.backends.cudnn.benchmark = True

# --- Unchanged Data Loading Section ---

path = 'CWRU_data'
num_classes = 12
percentage = 15
num_train_samples = (4800 * percentage) // 100
num_blocks = 9  # This will be adjusted for the new models
fs = 12000

csv_files = glob.glob(f'{path}*.csv')
dfs = [pd.read_csv(file) for file in csv_files]

class SingleCSVDataset(Dataset):
    def __init__(self, file_path, seq_len=1824, num_train_samples=1000, num_test_samples=400):
        filename = os.path.splitext(os.path.basename(file_path))[0].replace("_Sensor1", "")
        label_mapping = {
            "N": 0, "7BA": 1, "7IR": 2, "7OR": 3, "14BA": 4, "14IR": 5, "14OR": 6, "21BA": 7, "21IR": 8, "21OR": 9, "BA28": 10, "IR28": 11
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
    train_datasets = []
    test_datasets = []
    valid_datasets = []

    for file_path in csv_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        train_samples = num_train_samples if filename[0] != 'N' else 4800
        test_samples = num_test_samples if filename[0] != 'N' else 200
        dataset = SingleCSVDataset(file_path, seq_len, train_samples, test_samples)
        train_datasets.append(torch.utils.data.Subset(dataset, range(train_samples)))
        test_datasets.append(torch.utils.data.Subset(dataset, range(train_samples, train_samples + test_samples)))
        valid_datasets.append(torch.utils.data.Subset(dataset, range(train_samples + test_samples, train_samples + 2 * test_samples)))

    train_data = ConcatDataset(train_datasets)
    test_data = ConcatDataset(test_datasets)
    valid_data = ConcatDataset(valid_datasets)
    return train_data, test_data, valid_data


## Enhanced Self-Attention and Convolutional Blocks

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.out_proj = nn.Linear(in_channels, in_channels)

        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x shape: (batch, channels, length)
        batch_size, _, length = x.shape
        
        # Transpose to (batch, length, channels) for linear layers
        x = x.permute(0, 2, 1)

        # Reshape for multi-head processing: (batch, length, num_heads, head_dim)
        x_reshaped = x.view(batch_size, length, self.num_heads, self.head_dim)

        # Apply linear layers
        q = self.query(x_reshaped)
        k = self.key(x_reshaped)
        v = self.value(x_reshaped)
        
        # Transpose for attention calculation: (batch, num_heads, length, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v) # (batch, num_heads, length, head_dim)

        # Reshape and project back to original dimensions
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, length, self.in_channels)
        out = self.out_proj(attn_output)

        # Transpose back to (batch, channels, length) and add residual connection
        out = out.permute(0, 2, 1)
        return self.gamma * out + x.permute(0, 2, 1)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                  padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class EnhancedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size,
                                             padding='same', dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size,
                                             padding='same', dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.se = SqueezeExcitation(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.se(x)
        return x + residual

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1, expansion_factor=4):
        super().__init__()
        self.attention = MultiHeadSelfAttention(channels, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * expansion_factor, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * expansion_factor, channels, 1),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.layer_scale1 = nn.Parameter(torch.ones(channels) * 1e-6)
        self.layer_scale2 = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x):
        x_norm = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        attn_out = self.attention(x_norm)
        x = x + attn_out * self.layer_scale1.unsqueeze(0).unsqueeze(-1)
        x_norm = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out * self.layer_scale2.unsqueeze(0).unsqueeze(-1)
        return x

class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=12, embed_size=12, num_tcn_blocks=12,
                 num_transformer_blocks=8, channels=128, kernel_size=5, dropout=0.1,
                 output_length=1824):
        super().__init__()
        self.output_length = output_length
        self.channels = channels
        
        self.label_emb = nn.Embedding(num_classes, embed_size)
        
        initial_length = 114 # 1824 / (2^4)
        
        self.init_proj = nn.Linear(nz + embed_size, channels * initial_length)
        
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.LayerNorm([channels, initial_length * (2**(i+1))]),
                nn.GELU()
            ) for i in range(4) # Upsample 4 times: 114 -> 228 -> 456 -> 912 -> 1824
        ])
        
        self.tcn_blocks = nn.ModuleList([
            EnhancedTCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_tcn_blocks)
        ])
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channels, num_heads=8, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        self.output_conv = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], dim=1)
        x = self.init_proj(x)
        x = x.view(x.size(0), self.channels, -1)
        
        for upsample_layer in self.upsample_blocks:
            x = upsample_layer(x)
            
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
            
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
            
        x = self.output_conv(x)
        return x

## Enhanced Discriminator
class EnhancedDiscriminator(nn.Module):
    def __init__(self, num_classes=12, channels=64, kernel_size=5, dropout=0.1, num_tcn_blocks=6):
        super().__init__()
        
        self.input_convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(1, channels // 3, kernel_size=k, padding=k // 2))
            for k in [3, 5, 7]
        ])
        
        self.tcn_blocks = nn.ModuleList([
             nn.Sequential(
                spectral_norm(nn.Conv1d(channels if i > 0 else channels, channels, kernel_size, padding='same', dilation=2**i)),
                nn.LayerNorm([channels, 1824]),
                nn.LeakyReLU(0.2, inplace=True)
            ) for i in range(num_tcn_blocks)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.adv_output = nn.Linear(channels, 1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        # Multi-scale input processing
        input_features = [conv(x) for conv in self.input_convs]
        # Pad the feature map from kernel 3 and 7 to match kernel 5 (channels // 3)
        # Note: 64 // 3 = 21. Total channels = 21+21+21 = 63. We pad one channel.
        pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        x = torch.cat(input_features + [pad], dim=1)
        
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
            
        x = self.pool(x).squeeze(-1)
        validity = self.adv_output(x)
        label_pred = self.classifier(x)
        return validity, label_pred

## Teacher-Forcing Supervisor
class TeacherForcingSupervisor(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels // 4, 5, padding=2), nn.ReLU(),
            nn.Conv1d(channels // 4, channels // 2, 5, padding=2), nn.ReLU(),
            nn.Conv1d(channels // 2, channels, 5, padding=2),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, real, fake):
        real_features = self.encoder(real).squeeze(-1)
        fake_features = self.encoder(fake).squeeze(-1)
        return F.mse_loss(fake_features, real_features)

# --- Enhanced Training Function ---

def compute_gradient_penalty(netD, real_data, fake_data, labels):
    b_size = real_data.size(0)
    eps = torch.rand(b_size, 1, 1, device=real_data.device)
    interpolates = eps * real_data + (1 - eps) * fake_data
    interpolates.requires_grad_(True)
    d_interpolates, _ = netD(interpolates)
    grad_outputs = torch.ones(d_interpolates.size(), device=real_data.device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=grad_outputs, create_graph=True,
        retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(b_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

def train_enhanced_acwgan(dataloader, save_path=""):
    # Hyperparameters
    lr_g = 1e-4
    lr_d = 2e-4
    p_coeff = 10.0
    n_critic = 5
    epoch_num = 50
    nz = 100
    cls_coeff = 1.0
    teacher_coeff = 0.5
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize networks and supervisor
    netD = EnhancedDiscriminator(num_classes=num_classes).to(device)
    netG = Generator(nz=nz, num_classes=num_classes).to(device)
    supervisor = TeacherForcingSupervisor().to(device)
    
    # Check for saved models
    D_path = os.path.join(save_path, f"{num_train_samples}_wgan-True_{netG.__class__.__name__}-{netD.__class__.__name__}D_50.pth")
    G_path = os.path.join(save_path, f"{num_train_samples}_wgan-True_{netG.__class__.__name__}-{netD.__class__.__name__}G_50.pth")

    if os.path.exists(D_path) and os.path.exists(G_path):
        print("Loading saved models...")
        netD.load_state_dict(torch.load(D_path))
        netG.load_state_dict(torch.load(G_path))
    else:
        print("No saved models found. Initializing from scratch...")
    
    # Optimizers
    optimizerD = optim.AdamW(netD.parameters(), lr=lr_d, betas=(0.0, 0.9), weight_decay=0.01)
    optimizerG = optim.AdamW(netG.parameters(), lr=lr_g, betas=(0.0, 0.9), weight_decay=0.01)

    # Schedulers
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=epoch_num)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=epoch_num)

    class_criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        for step, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)
            b_size = real_data.size(0)

            # --- Train Discriminator ---
            netD.zero_grad()
            noise = torch.randn(b_size, nz, device=device)
            fake_data = netG(noise, labels).detach()

            d_real, class_logits_real = netD(real_data)
            d_fake, _ = netD(fake_data)

            loss_D_wasserstein = torch.mean(d_fake) - torch.mean(d_real)
            loss_D_class = class_criterion(class_logits_real, labels) * cls_coeff
            gp = compute_gradient_penalty(netD, real_data, fake_data, labels) * p_coeff
            
            loss_D = loss_D_wasserstein + gp + loss_D_class
            loss_D.backward()
            optimizerD.step()

            # --- Train Generator ---
            if step % n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, nz, device=device)
                fake_data = netG(noise, labels)

                d_fake, class_logits_fake = netD(fake_data)
                
                loss_G_wasserstein = -torch.mean(d_fake)
                loss_G_class = class_criterion(class_logits_fake, labels) * cls_coeff
                loss_G_teacher = supervisor(real_data, fake_data) * teacher_coeff

                loss_G = loss_G_wasserstein + loss_G_class + loss_G_teacher
                loss_G.backward()
                optimizerG.step()
            
            if step % 20 == 0:
                print(f"[Epoch {epoch}/{epoch_num}][Step {step}/{len(dataloader)}] "
                      f"Loss_D: {loss_D.item():.4f} (W: {loss_D_wasserstein.item():.4f}, GP: {gp.item():.4f}, Cls: {loss_D_class.item():.4f}) | "
                      f"Loss_G: {loss_G.item():.4f} (W: {loss_G_wasserstein.item():.4f}, Cls: {loss_G_class.item():.4f}, Teach: {loss_G_teacher.item():.4f})")

        scheduler_D.step()
        scheduler_G.step()

        torch.save(netG.state_dict(), f"{save_path}{num_train_samples}_wgan-True_{netG.__class__.__name__}-{netD.__class__.__name__}G_50.pth")
        torch.save(netD.state_dict(), f"{save_path}{num_train_samples}_wgan-True_{netG.__class__.__name__}-{netD.__class__.__name__}D_50.pth")
        print(f"Epoch {epoch} | Models saved at {save_path}")

    return netG, netD

# --- Unchanged Main Execution & Evaluation ---

if __name__ == "__main__":
    csv_folder = f"{path}"
    
    # Load dataset
    train_data, test_data, valid_data = load_all_data(csv_folder, num_train_samples=num_train_samples, num_test_samples=200)
    print(f"Total training samples: {len(train_data)}")
    print(f"Total testing samples: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

    # Training
    start_time = time.time()
    train_enhanced_acwgan(train_loader)
    training_time = time.time() - start_time
    np.save(f"{num_train_samples}_training_time.npy", training_time)

    # Evaluation Section
    # Helper functions
    def compute_fft(signal):
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(fft_vals), d=1/fs)
        pos_idx = freqs > 0
        return np.abs(fft_vals[pos_idx]), freqs[pos_idx]

    def load_generator(GeneratorModel, num_classes, weight_path, nz=100, device='cpu'):
        netG = GeneratorModel(nz=nz, num_classes=num_classes).to(device)
        netG.load_state_dict(torch.load(weight_path, map_location=device))
        netG.eval()
        return netG

    def generate_sequences(generator, label, n_samples=1, nz=100, device='cpu'):
        noise = torch.randn(n_samples, nz, device=device)
        label_tensor = torch.full((n_samples,), label, dtype=torch.long, device=device)
        with torch.no_grad():
            return generator(noise, label_tensor).cpu().numpy().squeeze()

    def compute_metrics(original, generated):
        pearson_corr, _ = pearsonr(original, generated)
        cosine_sim = cosine_similarity(original.reshape(1, -1), generated.reshape(1, -1))[0][0]
        # Adding epsilon to prevent division by zero in entropy calculation
        original_norm = original / original.sum() + 1e-9
        generated_norm = generated / generated.sum() + 1e-9
        kl_div = entropy(original_norm, generated_norm)
        t_stat, p_value = ttest_ind(original, generated)
        return pearson_corr, cosine_sim, kl_div, t_stat, p_value

    def plot_ffts_all(original_freqs, original_mags, generated_freqs, generated_mags, titles):
        plt.figure(figsize=(12, 16))
        for i in range(min(10, len(titles))):
            plt.subplot(5, 2, i + 1)
            plt.plot(original_freqs[i], original_mags[i], label="Original FFT", color='black')
            plt.plot(generated_freqs[i], generated_mags[i], label="Generated FFT", color='red', linestyle='--')
            plt.title(titles[i])
            plt.xlabel("Frequency (Hz)")
            plt.xlim(0, fs // 8)
            plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.savefig(f"{num_train_samples}_fft_comparison.png")
        plt.show()

    # Main evaluation logic
    weight_path = f'./{num_train_samples}_wgan-Generator-EnhancedDiscriminatorG_50.pth'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    GeneratorModel = Generator

    label_mapping = {
        "N": 0, "7BA": 1, "7IR": 2, "7OR": 3, "14BA": 4, "14IR": 5, "14OR": 6, 
        "21BA": 7, "21IR": 8, "21OR": 9, "BA28": 10, "IR28": 11
    }
    
    results_csv_path = f"{num_train_samples}_generation_results_SOTA.csv"
    original_freqs_all, original_mags_all, generated_freqs_all, generated_mags_all = [], [], [], []

    with open(results_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Avg Pearson", "Avg Cosine", "Avg KL", "Avg P-value"])

        generator = load_generator(GeneratorModel, num_classes, weight_path, device=device)

        for file_name, label_value in label_mapping.items():
            pearson_corrs, cosine_sims, kl_divs, p_values = [], [], [], []
            generated_samples_for_csv = []

            for _ in range(100):
                indexes = [idx for idx, (_, label) in enumerate(test_data) if label == label_value]
                random_index = np.random.choice(indexes)
                original_signal = test_data[random_index][0].numpy().flatten()
                original_mag, original_freq = compute_fft(original_signal)

                generated_signal = generate_sequences(generator, label_value, n_samples=1, device=device)
                if generated_signal.ndim > 1:
                    generated_signal = generated_signal[0]
                
                generated_samples_for_csv.append(generated_signal)
                generated_mag, generated_freq = compute_fft(generated_signal)

                # Ensure mags have the same length for comparison
                min_len = min(len(original_mag), len(generated_mag))
                original_mag, generated_mag = original_mag[:min_len], generated_mag[:min_len]

                pearson_corr, cosine_sim, kl_div, _, p_value = compute_metrics(original_mag, generated_mag)
                pearson_corrs.append(pearson_corr)
                cosine_sims.append(cosine_sim)
                kl_divs.append(kl_div)
                p_values.append(p_value)

            avg_pearson = np.mean(pearson_corrs)
            avg_cosine = np.mean(cosine_sims)
            avg_kl = np.mean(kl_divs)
            avg_p_value = np.mean(p_values)

            writer.writerow([file_name, avg_pearson, avg_cosine, avg_kl, avg_p_value])
            print(f"{file_name}: Pearson={avg_pearson:.4f}, Cosine={avg_cosine:.4f}, KL={avg_kl:.4f}, P-value={avg_p_value:.4f}")

            original_freqs_all.append(original_freq)
            original_mags_all.append(original_mag)
            generated_freqs_all.append(generated_freq)
            generated_mags_all.append(generated_mag)

            # Save generated samples
            generated_samples_arr = np.stack(generated_samples_for_csv, axis=1)
            gen_csv_path = f"{num_train_samples}_generated_{file_name}.csv"
            np.savetxt(gen_csv_path, generated_samples_arr, delimiter=",")
            print(f"Generated samples for {file_name} saved to {gen_csv_path}")

    plot_ffts_all(original_freqs_all, original_mags_all, generated_freqs_all, generated_mags_all, list(label_mapping.keys()))
    print(f"Results stored in {results_csv_path}")