import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy, wasserstein_distance

real = pd.read_csv("original_data.csv", header=None)[0].values
synthetic = pd.read_csv("local-llm/local-llm-data-v5.csv", header=None)[0].values.astype(float)


plt.figure(figsize=(16, 9))
plt.plot(real[:2400], label='Real', alpha=0.7)
plt.plot(synthetic[:2400], label='Synthetic Local LLM', alpha=0.7)
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
def compute_time_domain_features(signal):
    mean_val = np.mean(signal)
    std_dev = np.std(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    abs_mean = np.mean(np.abs(signal))
    peak_val = np.max(np.abs(signal))
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    var_val = np.var(signal)
    
    kurt_index = kurt_val / (rms**4 + 1e-8)
    peak_index = peak_val / (rms + 1e-8)
    waveform_index = rms / (abs_mean + 1e-8)
    pulse_index = peak_val / (abs_mean + 1e-8)

    return {
        'mean_value': mean_val,
        'standard_deviation': std_dev,
        'square_root_amplitude': rms,
        'absolute_mean_value': abs_mean,
        'peak_value': peak_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'variance': var_val,
        'kurtosis_index': kurt_index,
        'peak_index': peak_index,
        'waveform_index': waveform_index,
        'pulse_index': pulse_index
    }
def compute_frequency_domain_features(signal, fs):
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals)
    fft_freqs = np.fft.fftfreq(len(signal), d=1/fs)
    
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    spectrum = fft_mag[pos_mask]
    norm_spec = spectrum / (np.sum(spectrum) + 1e-8)

    centroid = np.sum(freqs * norm_spec)
    spread = np.sqrt(np.sum((freqs - centroid) ** 2 * norm_spec))
    rolloff = freqs[np.where(np.cumsum(norm_spec) >= 0.85)[0][0]]
    energy = np.sum(spectrum ** 2)
    peak = np.max(spectrum)
    flatness = np.exp(np.mean(np.log(spectrum + 1e-8))) / (np.mean(spectrum) + 1e-8)

    return {
        'frequency_mean_value': np.mean(spectrum),
        'frequency_variance': np.var(spectrum),
        'frequency_skewness': skew(spectrum),
        'frequency_kurtosis': kurtosis(spectrum),
        'frequency_standard_deviation': np.std(spectrum),
        'frequency_root_mean_square': np.sqrt(np.mean(spectrum ** 2)),
        'average_frequency': centroid,
        'gravity_frequency': centroid,  # synonym
        'regularity_degree': spread,
        'variation_parameter': spread / (centroid + 1e-8),
        'eighth_order_moment': np.mean(spectrum ** 8),
        'sixteenth_order_moment': np.mean(spectrum ** 16),
        'entropy': entropy(norm_spec),
        'spectral_rolloff_85': rolloff,
        'spectral_energy': energy,
        'spectral_peak': peak,
        'spectral_flatness': flatness
    }, freqs, spectrum
def print_comparison_stats(real_dict, synth_dict, domain="Time"):
    print(f"\n===== {domain} Domain Feature Comparison =====")
    for key in real_dict:
        real_val = real_dict[key]
        synth_val = synth_dict[key]
        diff = abs(real_val - synth_val)
        print(f"{key:30s} | Real: {real_val:>10.5f} | Synthetic: {synth_val:>10.5f} | Δ: {diff:>10.5f}")
real_time_stats = compute_time_domain_features(real)
synth_time_stats = compute_time_domain_features(synthetic)
print_comparison_stats(real_time_stats, synth_time_stats, domain="Time")
real_freq_stats, _, _ = compute_frequency_domain_features(real, fs)
synth_freq_stats, _, _ = compute_frequency_domain_features(synthetic, fs)
print_comparison_stats(real_freq_stats, synth_freq_stats, domain="Frequency")
