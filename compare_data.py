import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, entropy, wasserstein_distance
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import rbf_kernel
from scipy.interpolate import interp1d
real = pd.read_csv("data.csv", header=None)[0].values
synthetic = pd.read_csv("local-llm/data_generated.csv", header=None)[0].values.astype(float)
real = real[:48000]
synthetic = synthetic[:48000]   

plt.figure(figsize=(16, 9))
plt.plot(real[:1000], label='Real', alpha=0.7)
plt.plot(synthetic[:1000], label='Synthetic Local LLM', alpha=0.5)
plt.title("Time-Domain Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

#fft
fs = 25600
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
# def compute_time_domain_features(signal):
#     mean_val = np.mean(signal)
#     std_dev = np.std(signal)
#     rms = np.sqrt(np.mean(np.square(signal)))
#     abs_mean = np.mean(np.abs(signal))
#     peak_val = np.max(np.abs(signal))
#     skew_val = skew(signal)
#     kurt_val = kurtosis(signal)
#     var_val = np.var(signal)
    
#     kurt_index = kurt_val / (rms**4 + 1e-8)
#     peak_index = peak_val / (rms + 1e-8)
#     waveform_index = rms / (abs_mean + 1e-8)
#     pulse_index = peak_val / (abs_mean + 1e-8)

#     return {
#         'mean_value': mean_val,
#         'standard_deviation': std_dev,
#         'square_root_amplitude': rms,
#         'absolute_mean_value': abs_mean,
#         'peak_value': peak_val,
#         'skewness': skew_val,
#         'kurtosis': kurt_val,
#         'variance': var_val,
#         'kurtosis_index': kurt_index,
#         'peak_index': peak_index,
#         'waveform_index': waveform_index,
#         'pulse_index': pulse_index
#     }
# def compute_frequency_domain_features(signal, fs):
#     fft_vals = np.fft.fft(signal)
#     fft_mag = np.abs(fft_vals)
#     fft_freqs = np.fft.fftfreq(len(signal), d=1/fs)
    
#     pos_mask = fft_freqs > 0
#     freqs = fft_freqs[pos_mask]
#     spectrum = fft_mag[pos_mask]
#     norm_spec = spectrum / (np.sum(spectrum) + 1e-8)

#     centroid = np.sum(freqs * norm_spec)
#     spread = np.sqrt(np.sum((freqs - centroid) ** 2 * norm_spec))
#     rolloff = freqs[np.where(np.cumsum(norm_spec) >= 0.85)[0][0]]
#     energy = np.sum(spectrum ** 2)
#     peak = np.max(spectrum)
#     flatness = np.exp(np.mean(np.log(spectrum + 1e-8))) / (np.mean(spectrum) + 1e-8)

#     return {
#         'frequency_mean_value': np.mean(spectrum),
#         'frequency_variance': np.var(spectrum),
#         'frequency_skewness': skew(spectrum),
#         'frequency_kurtosis': kurtosis(spectrum),
#         'frequency_standard_deviation': np.std(spectrum),
#         'frequency_root_mean_square': np.sqrt(np.mean(spectrum ** 2)),
#         'average_frequency': centroid,
#         'gravity_frequency': centroid,  # synonym
#         'regularity_degree': spread,
#         'variation_parameter': spread / (centroid + 1e-8),
#         'eighth_order_moment': np.mean(spectrum ** 8),
#         'sixteenth_order_moment': np.mean(spectrum ** 16),
#         'entropy': entropy(norm_spec),
#         'spectral_rolloff_85': rolloff,
#         'spectral_energy': energy,
#         'spectral_peak': peak,
#         'spectral_flatness': flatness
#     }, freqs, spectrum
# def print_comparison_stats(real_dict, synth_dict, domain="Time"):
#     print(f"\n===== {domain} Domain Feature Comparison =====")
#     for key in real_dict:
#         real_val = real_dict[key]
#         synth_val = synth_dict[key]
#         diff = abs(real_val - synth_val)
#         print(f"{key:30s} | Real: {real_val:>10.5f} | Synthetic: {synth_val:>10.5f} | Δ: {diff:>10.5f}")
# real_time_stats = compute_time_domain_features(real)
# synth_time_stats = compute_time_domain_features(synthetic)
# print_comparison_stats(real_time_stats, synth_time_stats, domain="Time")
# real_freq_stats, _, _ = compute_frequency_domain_features(real, fs)
# synth_freq_stats, _, _ = compute_frequency_domain_features(synthetic, fs)
# print_comparison_stats(real_freq_stats, synth_freq_stats, domain="Frequency")
def plot_kde(real, synthetic, labels=('Real', 'Synthetic'), title='KDE Plot of Signals'):
    plt.figure(figsize=(10, 5))
    
    sns.kdeplot(real, label=labels[0], fill=True, color='blue', linewidth=2, alpha=0.6)
    sns.kdeplot(synthetic, label=labels[1], fill=True, color='orange', linewidth=2, alpha=0.6)
    
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_kde(real, synthetic)

def interpolate_to_match(a, b):
    len_a, len_b = len(a), len(b)
    if len_a == len_b:
        return a, b
    elif len_a < len_b:
        x_old = np.linspace(0, 1, len_a)
        x_new = np.linspace(0, 1, len_b)
        a_interp = interp1d(x_old, a, kind='linear')(x_new)
        return a_interp, b
    else:
        x_old = np.linspace(0, 1, len_b)
        x_new = np.linspace(0, 1, len_a)
        b_interp = interp1d(x_old, b, kind='linear')(x_new)
        return a, b_interp

def pearson_corr(arr1, arr2):
    a, b = interpolate_to_match(arr1, arr2)
    return np.corrcoef(a, b)[0, 1]

def cosine_sim(arr1, arr2):
    a, b = interpolate_to_match(arr1, arr2)
    return 1 - cosine(a, b)

def kl_divergence(arr1, arr2, bins=100):
    p_hist, _ = np.histogram(arr1, bins=bins, density=True)
    q_hist, _ = np.histogram(arr2, bins=bins, density=True)
    p_hist += 1e-10
    q_hist += 1e-10
    return entropy(p_hist, q_hist)

def mmd(arr1, arr2, gamma=1.0, window_size=10000):
    a, b = interpolate_to_match(arr1, arr2)
    min_len = min(len(a), len(b))
    
    a = a[:min_len]
    b = b[:min_len]

    num_windows = min_len // window_size
    mmd_values = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size

        window_a = a[start:end].reshape(-1, 1)
        window_b = b[start:end].reshape(-1, 1)

        K_xx = rbf_kernel(window_a, window_a, gamma=gamma)
        K_yy = rbf_kernel(window_b, window_b, gamma=gamma)
        K_xy = rbf_kernel(window_a, window_b, gamma=gamma)

        mmd_val = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
        mmd_values.append(mmd_val)

    return np.mean(mmd_values)


def compute_all_metrics(signal_dict, fs=12000):
    keys = list(signal_dict.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            name1, name2 = keys[i], keys[j]
            arr1, arr2 = signal_dict[name1], signal_dict[name2]

            print(f"\nComparing real vs synthetic:")
            print(f"Lengths: {len(arr1)} vs {len(arr2)}")

            print(f"Pearson Correlation     : {pearson_corr(arr1, arr2):.4f}")
            print(f"Cosine Similarity       : {cosine_sim(arr1, arr2):.4f}")
            print(f"KL Divergence (hist)    : {kl_divergence(arr1, arr2):.4f}")
            print(f"Maximum Mean Discrepancy: {mmd(arr1, arr2):.6f}")
signals={
    'real': real[:48000],  
    'synthetic': synthetic[:48000]
}
compute_all_metrics(signals, fs=12000)
