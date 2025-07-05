import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_amplitude_data_from_csv(csv_file_path, window_size=5, sensor_name="amplitude"):
    df = pd.read_csv(csv_file_path, header=None)
    
    raw_sensor_data = []
    
    for i in range(0, len(df) - window_size + 1, window_size):
        values = df.iloc[i:i+window_size, 0].tolist()
        values = [float(v) for v in values if pd.notna(v)]
        
        if values:
            window_data = {sensor_name: values}
            raw_sensor_data.append(window_data)
    
    augmented_data = []
    for sample in raw_sensor_data:
        original_values = sample[sensor_name]
        
        for noise_level in [0.01, 0.05, 0.1]:
            noisy_values = [v + np.random.normal(0, noise_level * abs(v)) for v in original_values]
            augmented_data.append({sensor_name: noisy_values})
        
        for scale in [0.9, 1.1, 1.2]:
            scaled_values = [v * scale for v in original_values]
            augmented_data.append({sensor_name: scaled_values})
    
    raw_sensor_data.extend(augmented_data)
    return raw_sensor_data

class ImprovedSensorTokenizer:
    def __init__(self, num_bins=100):
        self.num_bins = num_bins
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        
        self.PAD_TOKEN = "<PAD>"
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"
        self.SEP_TOKEN = "<SEP>"
        
    def build_vocab(self, sensor_data):
        tokens = set()
        tokens.update([self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.SEP_TOKEN])
        
        all_values = []
        for sample in sensor_data:
            for sensor_name, values in sample.items():
                tokens.add(f"SENSOR_{sensor_name}")
                all_values.extend([float(v) for v in values])
        
        if all_values:
            self.min_val = min(all_values)
            self.max_val = max(all_values)
            self.bin_size = (self.max_val - self.min_val) / self.num_bins if self.max_val != self.min_val else 1.0
            
            for i in range(self.num_bins):
                tokens.add(f"BIN_{i}")
        else:
            self.min_val = 0
            self.max_val = 1
            self.bin_size = 1.0
        
        sorted_tokens = sorted(list(tokens))
        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def value_to_bin(self, value):
        if self.bin_size == 0:
            return 0
        bin_idx = min(int((float(value) - self.min_val) / self.bin_size), self.num_bins - 1)
        return max(0, bin_idx)
        
    def encode(self, sensor_sample):
        tokens = [self.START_TOKEN]
        
        for sensor_name, values in sensor_sample.items():
            tokens.append(f"SENSOR_{sensor_name}")
            
            for value in values:
                bin_idx = self.value_to_bin(value)
                tokens.append(f"BIN_{bin_idx}")
            
            tokens.append(self.SEP_TOKEN)
        
        tokens.append(self.END_TOKEN)
        
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.PAD_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids):
        tokens = [self.reverse_vocab.get(idx, self.PAD_TOKEN) for idx in token_ids]
        
        sensor_data = {}
        current_sensor = None
        current_values = []
        
        for token in tokens:
            if token.startswith("SENSOR_"):
                if current_sensor and current_values:
                    sensor_data[current_sensor] = current_values
                current_sensor = token.replace("SENSOR_", "")
                current_values = []
            elif token.startswith("BIN_"):
                try:
                    bin_idx = int(token.replace("BIN_", ""))
                    value = self.min_val + (bin_idx + 0.5) * self.bin_size
                    current_values.append(value)
                except ValueError:
                    continue
            elif token == self.SEP_TOKEN:
                if current_sensor and current_values:
                    sensor_data[current_sensor] = current_values
                    current_sensor = None
                    current_values = []
        
        if current_sensor and current_values:
            sensor_data[current_sensor] = current_values
        
        return sensor_data

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        if x.is_cuda:
            mask = mask.cuda()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SmallSensorLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, 
                 d_ff=256, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        if seq_len > self.max_seq_len:
            x = x[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        token_emb = self.token_embedding(x)
        
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        x = self.dropout(token_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class SensorDataset(Dataset):
    def __init__(self, sensor_data, tokenizer, max_length=128):
        self.data = sensor_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        token_ids = self.tokenizer.encode(sample)
        
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            pad_id = self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]
            token_ids.extend([pad_id] * (self.max_length - len(token_ids)))
        
        token_ids = [max(0, min(id, self.tokenizer.vocab_size - 1)) for id in token_ids]
        
        return torch.tensor(token_ids, dtype=torch.long)

def prepare_sensor_data(raw_data):
    processed_data = []
    
    for sample in raw_data:
        processed_sample = {}
        for sensor_name, values in sample.items():
            if isinstance(values, (list, np.ndarray)):
                processed_sample[sensor_name] = list(values)
            else:
                processed_sample[sensor_name] = [values]
        processed_data.append(processed_sample)
    
    return processed_data

def train_model(model, dataloader, tokenizer, epochs=15, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.PAD_TOKEN])
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            batch = torch.clamp(batch, 0, tokenizer.vocab_size - 1)
            
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            if input_ids.size(1) == 0:
                continue
                
            optimizer.zero_grad()
            
            try:
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

def generate_synthetic_data(model, tokenizer, prompt_sample=None, 
                          num_samples=5, max_length=100, temperature=0.9):
    device = next(model.parameters()).device
    model.eval()
    
    synthetic_samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            if prompt_sample:
                input_ids = tokenizer.encode(prompt_sample)
            else:
                input_ids = [tokenizer.vocab[tokenizer.START_TOKEN]]
            
            input_tensor = torch.tensor([input_ids], device=device)
            
            for _ in range(max_length - len(input_ids)):
                if input_tensor.size(1) >= model.max_seq_len:
                    break
                    
                logits = model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == tokenizer.vocab[tokenizer.END_TOKEN]:
                    break
                
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token]], device=device)
                ], dim=1)
            
            generated_ids = input_tensor[0].cpu().tolist()
            synthetic_sample = tokenizer.decode(generated_ids)
            
            if synthetic_sample:
                synthetic_samples.append(synthetic_sample)
    
    return synthetic_samples

if __name__ == "__main__":
    csv_file_path = "original_data.csv"
    raw_sensor_data = load_amplitude_data_from_csv(csv_file_path, window_size=24)
    
    print(f"Loaded {len(raw_sensor_data)} samples")
    
    sensor_data = prepare_sensor_data(raw_sensor_data)
    
    tokenizer = ImprovedSensorTokenizer(num_bins=100)
    tokenizer.build_vocab(sensor_data)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    dataset = SensorDataset(sensor_data, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = SmallSensorLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=5,
        d_ff=128,
        max_seq_len=128,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Starting training...")
    train_model(model, dataloader, tokenizer, epochs=30, lr=0.001)
    
    print("\nGenerating synthetic sensor data...")
    synthetic_data = generate_synthetic_data(
        model, tokenizer,
        num_samples=100,
        max_length=50,
        temperature=0.8
    )
    
    def save_synthetic_data_to_csv(samples, filename="local-llm-data.csv"):
        all_values = []
        for sample in samples:
            for sensor_name, values in sample.items():
                all_values.extend(values)
            df = pd.DataFrame(all_values)
        df.to_csv(filename, index=False, header=False)
        print(f"Saved {len(all_values)} values to {filename}")
    save_synthetic_data_to_csv(synthetic_data)