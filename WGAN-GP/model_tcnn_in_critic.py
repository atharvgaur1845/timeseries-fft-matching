import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class CWRUDataset(Dataset):
    def __init__(self, csv_file, label, max_samples=None, window_size=1824, stride=36):
        data = pd.read_csv(csv_file)
        signal = data.iloc[:, 0].values.astype(np.float32)
        self.label = label
        self.sequences = []
        self.scalers = []
        
        for start in range(0, len(signal) - window_size + 1, stride):
            if max_samples is not None and len(self.sequences) >= max_samples:
                break
            window = signal[start:start + window_size]
            min_val = np.min(window)
            max_val = np.max(window)

            if max_val - min_val < 1e-7:
                normalized = np.zeros_like(window)
            else:
                normalized = 2 * (window - min_val) / (max_val - min_val) - 1
                
            self.sequences.append(normalized)
            self.scalers.append((min_val, max_val))
        
        self.sequences = np.array(self.sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]).unsqueeze(0),
                torch.tensor(self.label, dtype=torch.long),
                self.scalers[idx])
class GeneratorCNN(nn.Module):
    def __init__(self, nz=100, num_classes=4, embed_size=10):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_size)
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz + embed_size, 512, 114, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)  
        gen_input = torch.cat((z, label_embedding), dim=1)  
        gen_input = gen_input.unsqueeze(2) 
        x = self.main(gen_input)
        return x
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
class DiscriminatorTCN(nn.Module):
    def __init__(self, num_classes=4, num_blocks=9, channels=64, kernel_size=5, dropout=0.05):
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

def gradient_penalty(discriminator, real_data, fake_data, labels, device, lambda_gp=10):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    
    min_length = min(real_data.size(2), fake_data.size(2))
    real_cropped = real_data[:, :, :min_length]
    fake_cropped = fake_data[:, :, :min_length]
    
    interpolated = epsilon * real_cropped + (1 - epsilon) * fake_cropped
    interpolated.requires_grad_(True)
    
    critic_output, _ = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    
    return penalty

def generator_loss(discriminator, fake_data, fake_labels, lambda_cls=1.0):
    critic_output, class_output = discriminator(fake_data)
    
    wasserstein_loss = -critic_output.mean()
    
    classification_loss = F.cross_entropy(class_output, fake_labels)
    
    return wasserstein_loss + lambda_cls * classification_loss

def discriminator_loss(discriminator, real_data, fake_data, real_labels, fake_labels, device, lambda_gp=10, lambda_cls=1.0):
    real_critic, real_class = discriminator(real_data)
    fake_critic, fake_class = discriminator(fake_data.detach())
    
    wasserstein_loss = fake_critic.mean() - real_critic.mean()
    gp = gradient_penalty(discriminator, real_data, fake_data, real_labels, device, lambda_gp)
    classification_loss = F.cross_entropy(real_class, real_labels)
    
    return wasserstein_loss + gp + lambda_cls * classification_loss

def train(num_epochs=50, batch_size=64, lr=1e-4, lambda_gp=10, lambda_cls=1.0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    datasets = CWRUDataset('CWRU_data/N.csv', label=0, max_samples=4800)  

    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    
    print(f"Total training samples: {len(datasets)}")
    
    generator = GeneratorCNN(nz=100, num_classes=4, embed_size=10).to(device)
    discriminator = DiscriminatorTCN(num_classes=4, num_blocks=9, channels=64).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    g_losses = []
    d_losses = []
    
    print("training...")
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        for real_data, real_labels, scalers in dataloader:
            real_data = real_data.to(device)
            real_labels = real_labels.to(device)
            current_batch_size = real_data.size(0)
            
            for _ in range(5):
                noise = torch.randn(current_batch_size, 100, device=device)
                #fake_labels = torch.randint(0, 4, (current_batch_size,), device=device)
                fake_labels = torch.zeros(current_batch_size, dtype=torch.long, device=device)
                fake_data = generator(noise, fake_labels)
                
                d_loss = discriminator_loss(discriminator, real_data, fake_data,
                                          real_labels, fake_labels, device, lambda_gp, lambda_cls)
                
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                
                epoch_d_loss += d_loss.item()
            
            noise = torch.randn(current_batch_size, 100, device=device)
            fake_labels = torch.randint(0, 4, (current_batch_size,), device=device)
            fake_data = generator(noise, fake_labels)
            
            g_loss = generator_loss(discriminator, fake_data, fake_labels, lambda_cls)
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            num_batches += 1
        
        g_losses.append(epoch_g_loss / num_batches)
        d_losses.append(epoch_d_loss / (num_batches * 5))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - G_Loss: {g_losses[-1]:.4f}, D_Loss: {d_losses[-1]:.4f}")
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_noise = torch.randn(1, 100, device=device)
                test_labels = torch.tensor([0], device=device)  
                test_output = generator(test_noise, test_labels)
                print(f"  Generated shape: {test_output.shape}, Std: {test_output.std().item():.4f}")
    
    return generator, discriminator, g_losses, d_losses

def denormalize_data(normalized_data, min_val, max_val):
    return (normalized_data + 1) * (max_val - min_val) / 2 + min_val

def save_synthetic_samples(generator, num_samples, scalers, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    device = next(generator.parameters()).device
    
    generator.eval()
    with torch.no_grad():
        for class_label in range(1):  
            for i in range(num_samples // 1):
                noise = torch.randn(1, 100, device=device)
                label = torch.tensor([class_label], device=device)
                
                fake_sample = generator(noise, label)
                sample_data = fake_sample.squeeze().cpu().numpy()
                
                scaler = scalers[i % len(scalers)]
                denormalized = denormalize_data(sample_data, scaler[0], scaler[1])
                
                df = pd.DataFrame(denormalized, columns=[None])
                df.to_csv(f"{output_dir}/synthetic_class_{class_label}_sample_{i}.csv", index=False)
    
    print(f"Saved {num_samples} synthetic samples to {output_dir}/")
if __name__ == "__main__":
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator, discriminator, g_losses, d_losses = train(
        num_epochs=50,
        batch_size=64,
        lr=1e-4,
        lambda_gp=10,
        lambda_cls=1.0
    )
    
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    
    datasets = CWRUDataset('CWRU_data/N.csv', label=0, max_samples=4800),
        
    all_scalers = []
    for dataset in datasets:
        all_scalers.extend(dataset.scalers)
    
    save_synthetic_samples(
        generator=generator,
        num_samples=10,  
        scalers=all_scalers,
        output_dir="synthetic_data_tcn_in_critic"
    )
    
    
    print("Training completed successfully!")
