import pandas as pd
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

path = 'CWRU_data'
num_classes = 10
percentage = 10
num_train_samples = (4800 * percentage) // 100
num_blocks = 9
fs = 12000

csv_files = glob.glob(f'{path}*.csv')
dfs = [pd.read_csv(file) for file in csv_files]

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset

class SingleCSVDataset(Dataset):
    def __init__(self, file_path, seq_len=1824, num_train_samples=1000, num_test_samples=400):
        filename = os.path.splitext(os.path.basename(file_path))[0].replace("_Sensor1", "")
        label_mapping = {
            "N": 0, "7BA": 1, "7IR": 2, "7OR": 3, "14BA": 4, "14IR": 5, "14OR": 6, "21BA": 7, "21IR": 8, "21OR": 9
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same", dilation=dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1,2)).transpose(1,2)
        x = self.relu(x)
        x = self.dropout(x)
        return x + res

# ---------- New Generator ----------
class GeneratorTCN(nn.Module):
    def __init__(self, 
                 nz=100, 
                 num_classes=10, 
                 embed_size=10, 
                 num_blocks=9, 
                 channels=64, 
                 kernel_size=5, 
                 dropout=0.05, 
                 output_length=1824):
        super().__init__()
        self.output_length = output_length
        self.num_classes = num_classes
        self.channels = channels
        
        # Label embedding for conditional generation
        self.label_emb = nn.Embedding(num_classes, embed_size)
        
        # Fully connected to expand noise+label to (channels, output_length)
        self.fc = nn.Linear(nz + embed_size, channels * output_length)
        
        # TCN blocks (no up/downsampling, just same-length residual blocks)
        self.tcn_layers = nn.Sequential(*[
            TCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        
        # Final conv to map channels → 1 output
        self.to_output = nn.Conv1d(channels, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        # Get label embeddings
        label_embedding = self.label_emb(labels)  # (batch_size, embed_size)
        
        # Concatenate noise with labels
        x = torch.cat([z, label_embedding], dim=1)  # (batch_size, nz + embed_size)
        
        # Expand to (batch, channels, length)
        x = self.fc(x)  # (batch_size, channels * output_length)
        x = x.view(-1, self.channels, self.output_length)  # (batch_size, channels, output_length)
        
        # Pass through TCN layers
        x = self.tcn_layers(x)
        
        # Project to 1 channel output
        x = self.to_output(x)
        return self.tanh(x)  # (batch, 1, output_length)

class DiscriminatorTCN(nn.Module):
    def __init__(self, num_classes=10, num_blocks=9, channels=64, kernel_size=5, dropout=0.05):
        super().__init__()
        self.initial_conv = nn.Conv1d(1, channels, kernel_size=1)
        self.tcn_layers = nn.Sequential(*[TCNBlock(channels, channels, kernel_size, dilation=2**i, dropout=dropout) for i in range(num_blocks)])
        self.flatten = nn.AdaptiveAvgPool1d(1)
        self.adv_output = nn.Linear(channels, 1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.tcn_layers(x)
        x = self.flatten(x).squeeze(2)
        validity = self.adv_output(x)
        label_pred = self.classifier(x)
        return validity, label_pred

import torch.nn as nn

def weights_init(m):
    """ Custom weights initialization """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
import time

torch.backends.cudnn.benchmark = True

# Hyperparameters
p_coeff = 10
n_critic = 5
clip_value = 0.01
lr = 1e-4
epoch_num = 1
batch_size = 32
nz = 100
cls_coeff = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Selection
use_wgan_gp = True
losses_D, losses_G, losses_class = [], [], []

Discriminator = DiscriminatorTCN
Generator = GeneratorTCN  # Updated to use new GeneratorTCN

def compute_gradient_penalty(netD, real_data, fake_data):
    """ Computes the gradient penalty for WGAN-GP """
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
    gp = p_coeff * ((grad_norm - 1) ** 2).mean()
    return gp

def train_acwgan(dataloader, save_path=""):
    global losses_D, losses_G, losses_class

    # Initialize networks
    netD = Discriminator(num_classes=num_classes, num_blocks=num_blocks).to(device)
    netG = Generator(nz, num_classes=num_classes).to(device)

    # Check if saved models exist and load them
    D_path = os.path.join(save_path, f"{num_train_samples}_wgan-True_{netG.__class__.__name__}-{netD.__class__.__name__}D_50.pth")
    G_path = os.path.join(save_path, f"{num_train_samples}_wgan-True_{netG.__class__.__name__}-{netD.__class__.__name__}G_50.pth")

    if os.path.exists(D_path) and os.path.exists(G_path):
        print("Loading saved models...")
        netD.load_state_dict(torch.load(D_path))
        netG.load_state_dict(torch.load(G_path))
    else:
        print("No saved models found. Initializing from scratch...")
        netD.apply(weights_init)
        netG.apply(weights_init)

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))

    for epoch in range(epoch_num):
        for step, (real_data, labels) in enumerate(dataloader):
            real_data, labels = real_data.to(device), labels.to(device)
            b_size = real_data.size(0)

            netD.zero_grad()

            # Generate fake data conditioned on labels
            noise = torch.randn(b_size, nz, device=device)
            fake_data = netG(noise, labels).detach()

            # Get Discriminator predictions
            d_real, class_logits_real = netD(real_data)
            d_fake, _ = netD(fake_data)

            # Wasserstein loss for discriminator
            loss_D_wasserstein = torch.mean(d_fake) - torch.mean(d_real)

            # Classification loss
            class_criterion = nn.CrossEntropyLoss()
            loss_D_class = (class_criterion(class_logits_real, labels) * cls_coeff)

            if use_wgan_gp:
                # Gradient penalty
                gp = compute_gradient_penalty(netD, real_data, fake_data)
                # Total Discriminator loss
                loss_D = loss_D_wasserstein + gp + loss_D_class
            else:
                loss_D = loss_D_wasserstein + loss_D_class
                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            loss_D.backward()
            optimizerD.step()

            if step % n_critic == 0:
                netG.zero_grad()

                # Generate fake samples
                noise = torch.randn(b_size, nz, device=device)
                fake_data = netG(noise, labels)

                # Discriminator output for fake samples
                d_fake, class_logits_fake = netD(fake_data)

                # Generator loss (maximize realness of fake samples)
                loss_G_wasserstein = -torch.mean(d_fake)

                # Generator should produce class-conditioned samples
                loss_G_class = (class_criterion(class_logits_fake, labels) * cls_coeff)

                # Total Generator loss
                loss_G = loss_G_wasserstein + loss_G_class

                loss_G.backward()
                optimizerG.step()

                # Logging losses
                losses_D.append(loss_D.item())
                losses_G.append(loss_G.item())
                losses_class.append(loss_D_class.item())

                if step % 5 == 0:
                    if use_wgan_gp:
                        print(f"[Epoch {epoch}/{epoch_num}][Step {step}/{len(dataloader)}] "
                              f"Loss_D: {loss_D.item():.4f} (W: {loss_D_wasserstein.item():.4f}, GP: {gp.item():.4f}, Cls: {loss_D_class.item():.4f}) | "
                              f"Loss_G: {loss_G.item():.4f} (W: {loss_G_wasserstein.item():.4f}, Cls: {loss_G_class.item():.4f})")
                    else:
                        print(f"[Epoch {epoch}/{epoch_num}][Step {step}/{len(dataloader)}] "
                              f"Loss_D: {loss_D.item():.4f} (W: {loss_D_wasserstein.item():.4f}, Cls: {loss_D_class.item():.4f}) | "
                              f"Loss_G: {loss_G.item():.4f} (W: {loss_G_wasserstein.item():.4f}, Cls: {loss_G_class.item():.4f})")

        # Save models after every epoch
        torch.save(netG.state_dict(), f"{save_path}{num_train_samples}_wgan-{use_wgan_gp}_{netG.__class__.__name__}-{netD.__class__.__name__}G_50.pth")
        torch.save(netD.state_dict(), f"{save_path}{num_train_samples}_wgan-{use_wgan_gp}_{netG.__class__.__name__}-{netD.__class__.__name__}D_50.pth")
        print(f"Epoch {epoch} | Models saved at {save_path}")

    return netG, netD

# Example usage:
if __name__ == "__main__":
    csv_folder = f"{path}"

    # Load dataset and create DataLoader
    train_data, test_data, valid_data = load_all_data(csv_folder, num_train_samples=num_train_samples, num_test_samples=200)
    print(f"Total training samples: {len(train_data)}")
    print(f"Total testing samples: {len(test_data)}")
    print(f"Total validation samples: {len(valid_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    ''' Training & Save training time'''
    start_time = time.time()
    train_acwgan(train_loader)
    training_time = time.time() - start_time
    np.save(f"{num_train_samples}_training_time.npy", training_time)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, entropy, ttest_ind, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import csv

# Load and normalize data
def load_data(csv_path):
    values = pd.read_csv(csv_path).values.flatten().astype(np.float32)
    values = 2 * (values - values.min()) / (values.max() - values.min() + 1e-8) - 1
    return values

# Compute FFT
def compute_fft(signal):
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft_vals), d=1/fs)
    pos_idx = freqs > 0
    return np.abs(fft_vals[pos_idx]), freqs[pos_idx]

# Load Generator Model
def load_generator(GeneratorModel, num_classes, weight_path, nz=100, device='cpu'):
    netG = GeneratorModel(nz, num_classes=num_classes).to(device)
    netG.load_state_dict(torch.load(weight_path, map_location=device))
    netG.eval()
    return netG

# Generate synthetic sequences
def generate_sequences(generator, label, n_samples=1, nz=100, device='cpu'):
    noise = torch.randn(n_samples, nz, device=device)
    label_tensor = torch.full((n_samples,), label, dtype=torch.long, device=device)
    with torch.no_grad():
        return generator(noise, label_tensor).cpu().numpy().squeeze()

# Compute MMD
def compute_mmd(x, y, sigma=1.0):
    x, y = torch.tensor(x).unsqueeze(1), torch.tensor(y).unsqueeze(1)
    kernel = lambda a, b: torch.exp(-((a - b.T) ** 2) / (2 * sigma ** 2))
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()

# Compute all metrics
def compute_metrics(original, generated):
    pearson_corr, _ = pearsonr(original, generated)
    cosine_sim = cosine_similarity(original.reshape(1, -1), generated.reshape(1, -1))[0][0]
    kl_div = entropy(original / original.sum(), generated / generated.sum())
    mmd_val = compute_mmd(original, generated).item()
    t_stat, p_value = ttest_ind(original, generated)
    return pearson_corr, cosine_sim, kl_div, mmd_val, t_stat, p_value

def plot_ffts_all(original_freqs, original_mags, generated_freqs, generated_mags, titles):
    plt.figure(figsize=(12, 16))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.plot(original_freqs[i], original_mags[i], label="Original FFT", color='black')
        plt.plot(generated_freqs[i], generated_mags[i], label="Generated FFT", color='red', linestyle='--')
        plt.title(titles[i].replace("_Sensor", "").replace("30hz", ""))
        plt.xlabel("Frequency (in Hz)")
        plt.xlim(0, fs // 8)
        plt.ylabel("Magnitude (in m/s²)")
        plt.tight_layout()
    plt.show()

# Main execution
percentage = 10
num_train_samples = (4800 * percentage) // 100
weight_path = f'./{num_train_samples}_wgan-True_GeneratorTCN-DiscriminatorTCNG_50.pth'  # Updated filename
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GeneratorModel = GeneratorTCN  # Updated to use new GeneratorTCN

label_mapping = {
    "N": 0, "7BA": 1, "7IR": 2, "7OR1": 3, "14BA": 4, "14IR": 5, "14OR1": 6, "21BA": 7, "21IR": 8, "21OR1": 9
}

# Define CSV file path
results_csv_path = f"{num_train_samples}_generation_results.csv"
original_freqs_all, original_mags_all, generated_freqs_all, generated_mags_all = [], [], [], []

# Open CSV file and write header
with open(results_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Test Name", "Avg Pearson", "Avg Cosine", "Avg KL", "Avg MMD", "Avg GAN-Test P-value"])

    for file_name, label_value in label_mapping.items():
        pearson_corrs, cosine_sims, kl_divs, mmd_vals, p_values = [], [], [], [], []
        generated_samples = []

        # Ensure 100 different generated samples by using different noise each time
        generator = load_generator(GeneratorModel, num_classes, weight_path, device=device)

        for _ in range(100):
            indexes = [idx for idx, (_, label) in enumerate(test_data) if label == label_value]
            random_index = np.random.choice(indexes)
            original_signal = test_data[random_index][0].numpy().flatten()
            original_mag, original_freq = compute_fft(original_signal)

            # Generate a different sample each time by using new noise
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

        avg_pearson = np.mean(pearson_corrs)
        avg_cosine = np.mean(cosine_sims)
        avg_kl = np.mean(kl_divs)
        avg_mmd = np.mean(mmd_vals)
        avg_p_value = np.mean(p_values)

        writer.writerow([file_name, avg_pearson, avg_cosine, avg_kl, avg_mmd, avg_p_value])
        print(f"{file_name}: Pearson={avg_pearson}, Cosine={avg_cosine}, KL={avg_kl}, MMD={avg_mmd}, P-value={avg_p_value}")

        original_freqs_all.append(original_freq)
        original_mags_all.append(original_mag)
        generated_freqs_all.append(generated_freq)
        generated_mags_all.append(generated_mag)

        # Save generated samples for this class to a CSV (each column is a sample)
        generated_samples_arr = np.stack(generated_samples, axis=1)
        gen_csv_path = f"{num_train_samples}_generated_{file_name}.csv"
        np.savetxt(gen_csv_path, generated_samples_arr, delimiter=",")
        print(f"Generated samples for {file_name} saved to {gen_csv_path}")

    plot_ffts_all(original_freqs_all, original_mags_all, generated_freqs_all, generated_mags_all, list(label_mapping.keys()))
    print(f"Results stored in {results_csv_path}")
