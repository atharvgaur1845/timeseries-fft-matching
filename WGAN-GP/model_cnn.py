import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, window_size=1824, stride=36):
        data = pd.read_csv(csv_file)
        signal = data.iloc[:, 0].values.astype(np.float32)
        self.min_val = np.min(signal)
        self.max_val = np.max(signal)
        signal = 2 * (signal - self.min_val) / (self.max_val - self.min_val) - 1
        
        self.sequences = []
        for start in range(0, len(signal) - window_size + 1, stride):
            self.sequences.append(signal[start:start+window_size])
        self.sequences = np.array(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]).unsqueeze(0)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_length=1824):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        

        self.initial = nn.Linear(latent_dim, 512 * 16)
        
        self.low_freq_branch = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=32, stride=4, padding=14),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6),
            nn.ReLU(True),
        )
        
        self.high_freq_branch = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        self.combiner = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z):
        x_1 = self.initial(z)
        x_1 = x_1.view(x_1.size(0), 512, 16)
        
        low_freq_input = x_1[:, :256, :]
        high_freq_input = x_1[:, 256:, :]
        
        low_freq = self.low_freq_branch(low_freq_input)
        high_freq = self.high_freq_branch(high_freq_input)
        
        target_len = min(low_freq.size(2), high_freq.size(2))
        low_freq = F.interpolate(low_freq, size=target_len, mode='linear', align_corners=False)
        high_freq = F.interpolate(high_freq, size=target_len, mode='linear', align_corners=False)
        
        combined = torch.cat([low_freq, high_freq], dim=1)
        combined = F.interpolate(combined, size=self.output_length, mode='linear', align_corners=False)
        
        x = self.combiner(combined)
        return x

# class GeneratorAdaptive(nn.Module):
#     def __init__(self, latent_dim=100, output_length=1824):
#         super(GeneratorAdaptive, self).__init__()
#         self.latent_dim = latent_dim
#         self.output_length = output_length
        
#         self.start_size = output_length // (4 * 4 * 4)
#         if self.start_size < 1:
#             self.start_size = 1
        
#         self.initial = nn.Linear(latent_dim, 512* self.start_size)
        
#         self.net = nn.Sequential(
#             nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv1d(32, 1, kernel_size=7, padding=3),
#             nn.Tanh()
#         )

    # def forward(self, z):
    #     x = self.initial(z)
    #     x = x.view(x.size(0), 512, self.start_size)
    #     x = self.net(x)
        
    #     if x.size(2) != self.output_length:
    #         x = torch.nn.functional.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
        
    #     return x

class Critic(nn.Module):
    def __init__(self, input_length=1824):
        super(Critic, self).__init__()
        
        self.multi_scale_convs = nn.ModuleList([
            #high frequency
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=3),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=16, stride=4, padding=6),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=6),
                nn.LeakyReLU(0.2),
            ),
            #low frequency
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=32, stride=8, padding=12),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, kernel_size=32, stride=8, padding=12),
                nn.LeakyReLU(0.2),
            )
        ])
        
        self.feature_combiner = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        multi_scale_features = []
        
        for conv_block in self.multi_scale_convs:
            feature = conv_block(x)
            feature = F.adaptive_avg_pool1d(feature, 100)
            multi_scale_features.append(feature)
        
        combined = torch.cat(multi_scale_features, dim=1)
        x = self.feature_combiner(combined)
        return x
    
def gradient_penalty(critic, real, fake, device, lambda_gp=10):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device, requires_grad=True)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    prob_interpolated = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    return penalty

def generator_loss(critic_output, classification_loss=0, lambda_g=1.0):
    return -torch.mean(critic_output) + lambda_g * classification_loss

def critic_loss(critic, real_data, fake_data, device, lambda_gp=10):
    real_loss = -torch.mean(critic(real_data))
    fake_loss = torch.mean(critic(fake_data))
    gp = gradient_penalty(critic, real_data, fake_data, device, lambda_gp)
    return real_loss + fake_loss + gp
def denormalize_data(normalized_data, min_val, max_val):
    return (normalized_data + 1) * (max_val - min_val) / 2 + min_val

def save_generated_to_csv(generated_tensor, filename, min_val, max_val):
    arr = generated_tensor.detach().cpu().numpy().reshape(-1)
    arr = denormalize_data(arr, min_val, max_val)
    df = pd.DataFrame(arr, columns=["generated"])
    df.to_csv(filename, index=False)
    print(f"Generated synthetic data saved to {filename}")
def plot_critic_losses(critic_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(critic_losses, label='Critic Loss', color='blue', linewidth=1.5)
    plt.xlabel('Training Step')
    plt.ylabel('Critic Loss')
    plt.title('WGAN-GP Critic Loss During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_wgan_gp(csv_file, num_epochs=50, batch_size=64, lr=1e-4, lambda_gp=10, 
                  lambda_g=1.0, n_critic=5, latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    dataset = TimeSeriesDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Choose generator type
    # if use_adaptive:
    #     generator = GeneratorAdaptive(latent_dim=latent_dim).to(device)
    #     print("Using adaptive generator")
    # else:
    #     generator = Generator(latent_dim=latent_dim).to(device)
    #     print("Using fixed generator")
    generator = Generator(latent_dim=latent_dim).to(device)
    critic = Critic().to(device)
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizer_c = optim.Adam(critic.parameters(), lr=lr, betas=(0.9, 0.99))
    
    critic_losses = []
    generator_losses = []
    
    for epoch in range(num_epochs):
        epoch_critic_losses = []
        epoch_gen_losses = []
        
        for i, real_samples in enumerate(dataloader):
            real_samples = real_samples.to(device)
            current_batch_size = real_samples.size(0)
            
            for _ in range(n_critic):
                z = torch.randn(current_batch_size, latent_dim).to(device)
                fake_samples = generator(z).detach()
                
                loss_c = critic_loss(critic, real_samples, fake_samples, device, lambda_gp)
                
                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()
                
                epoch_critic_losses.append(loss_c.item())
            
            z = torch.randn(current_batch_size, latent_dim).to(device)
            fake_samples = generator(z)
            critic_output = critic(fake_samples)
            
            loss_g = generator_loss(critic_output, classification_loss=0, lambda_g=lambda_g)
            
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            epoch_gen_losses.append(loss_g.item())
        
        critic_losses.extend(epoch_critic_losses)
        generator_losses.extend(epoch_gen_losses)
        
        avg_critic_loss = np.mean(epoch_critic_losses)
        avg_gen_loss = np.mean(epoch_gen_losses)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Critic Loss: {avg_critic_loss:.4f}, Generator Loss: {avg_gen_loss:.4f}")
    print("\nGenerating synthetic time series...")
    generator.eval()
    with torch.no_grad():
        num_samples = 100
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_samples = generator(z)
        print(f"Generated samples shape: {generated_samples.shape}")
       
        
        non_zero_count = (generated_samples != 0).sum().item()
        total_count = generated_samples.numel()
        print(f"Non-zero values: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.1f}%)")
    
    save_generated_to_csv(generated_samples, 'WGAN-GP/data_model_cnn.csv',dataset.min_val, dataset.max_val)
    plot_critic_losses(critic_losses)
    plot_critic_losses(generator_losses)
    return generator, critic, critic_losses, generator_losses

# Usage example:
if __name__ == "__main__":
    csv_file = 'CWRU_data/N.csv'
    
    generator, critic, critic_losses, gen_losses = train_wgan_gp(
        csv_file=csv_file,
        num_epochs=30,
        batch_size=256,
        lr=1e-4,
        lambda_gp=10,
        lambda_g=1.0,
        n_critic=2,
        latent_dim=256
    )
    
    print("Training completed!")
