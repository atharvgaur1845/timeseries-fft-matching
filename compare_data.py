import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
real = pd.read_csv("local-llm-data-v1.csv", header=None)[0].values
synthetic = pd.read_csv("local-llm-data-v2.csv", header=None)[0].values.astype(float)


plt.figure(figsize=(14, 4))
plt.plot(real[:2000], label='Real', alpha=0.7)
plt.plot(synthetic[:2000], label='Synthetic Local LLM', alpha=0.7)
plt.title("Time-Domain Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

#fft
fs = 12000 
N = len(real)

# Compute FFTs
def compute_fft(signal, fs):
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), d=1/fs)
    pos_mask = fft_freqs > 0
    return fft_freqs[pos_mask], (np.abs(fft_vals) * 2 / len(signal))[pos_mask]

real_freqs, real_fft = compute_fft(real, fs)
synth_freqs, synth_fft = compute_fft(synthetic, fs)

# Plot
plt.figure(figsize=(16, 9))
plt.plot(real_freqs, real_fft, label="Real FFT", alpha=0.7)
plt.plot(synth_freqs, synth_fft, label="Synthetic FFT Local LLM", alpha=0.7)
plt.title("FFT Magnitude Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

#statistical comparison
from scipy.stats import wasserstein_distance

# Mean, std
print("Real Mean:", np.mean(real), "\nSynthetic Mean:", np.mean(synthetic))
print("Real Std:", np.std(real), "\nSynthetic Std:", np.std(synthetic))

# EMD
print("\nWasserstein Distance:", wasserstein_distance(real_fft, synth_fft))
