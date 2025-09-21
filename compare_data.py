import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr, ttest_ind
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_to_match(a, b):
    """Interpolate arrays to match lengths"""
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

def normalize_signal(signal):
    """Apply same normalization as model.py"""
    signal = np.array(signal, dtype=np.float32)
    min_val, max_val = signal.min(), signal.max()
    return 2 * (signal - min_val) / (max_val - min_val + 1e-8) - 1

def compute_fft(signal, fs=12000):
    """Compute FFT of signal - same as model.py - FIXED ORDER"""
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(fft_vals), d=1/fs)
    pos_idx = freqs > 0
    return freqs[pos_idx], np.abs(fft_vals[pos_idx])  # Return (frequency, magnitude)

def compute_mmd(x, y, sigma=1.0):
    """Compute MMD - same as model.py"""
    import torch
    x, y = torch.tensor(x).unsqueeze(1), torch.tensor(y).unsqueeze(1)
    kernel = lambda a, b: torch.exp(-((a - b.T) ** 2) / (2 * sigma ** 2))
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()

def compute_metrics_fft(original, generated):
    """Compute all metrics on FFT domain - same as model.py"""
    pearson_corr, _ = pearsonr(original, generated)
    cosine_sim = cosine_similarity(original.reshape(1, -1), generated.reshape(1, -1))[0][0]
    kl_div = entropy(original / original.sum(), generated / generated.sum())
    mmd_val = compute_mmd(original, generated).item()
    t_stat, p_value = ttest_ind(original, generated)
    return pearson_corr, cosine_sim, kl_div, mmd_val, t_stat, p_value

def evaluate_class_metrics_multicolumn(real_base_dir, synthetic_base_dir, class_names, max_length=1824, fs=12000):
    """Evaluate metrics for all classes with 100-column synthetic files using FFT domain comparison"""
    real_file_mapping = {
        'N': 'N.csv',
        '14BA': '14BA.csv', '14IR': '14IR.csv', '14OR': '14OR.csv',
        '7BA': '7BA.csv', '7IR': '7IR.csv', '7OR': '7OR.csv',
        '21BA': '21BA.csv', '21IR': '21IR.csv', '21OR': '21OR.csv','BA28': 'BA28.csv','IR28':'IR28.csv'
    }
    
    best_results = {}
    
    for class_name in class_names:
        print(f"Processing class: {class_name}")
        
        # Load real data
        real_file = real_file_mapping.get(class_name)
        real_path = os.path.join(real_base_dir, real_file)
        if not os.path.exists(real_path):
            print(f"  Real file not found: {real_path}")
            continue
            
        real_data = pd.read_csv(real_path, header=None)
        real_signal = real_data.iloc[:, 0].values[:max_length]
        
        # Apply same normalization as model.py
        real_signal = normalize_signal(real_signal)
        
        # Compute FFT of real signal - now returns (frequency, magnitude)
        real_freq, real_mag = compute_fft(real_signal, fs)
        
        # Load synthetic data file (assuming single CSV file with 100 columns)
        synthetic_file_name = f"{class_name}.csv"
        synthetic_path = os.path.join(synthetic_base_dir, synthetic_file_name)
        
        if not os.path.exists(synthetic_path):
            print(f"  Synthetic file not found: {synthetic_path}")
            continue
        
        try:
            # Load synthetic data (100 columns)
            synthetic_data = pd.read_csv(synthetic_path, header=None)
            print(f"  Loaded synthetic data shape: {synthetic_data.shape}")
            
            # Initialize best metrics for this class
            best_metrics = {
                'pearson': {'value': -np.inf, 'column': None, 'data': None},
                'cosine': {'value': -np.inf, 'column': None, 'data': None},
                'kl_div': {'value': np.inf, 'column': None, 'data': None},
                'mmd': {'value': np.inf, 'column': None, 'data': None}
            }
            
            # Process all columns for this class
            for col_idx in tqdm(range(synthetic_data.shape[1]), desc=f"  Processing {class_name} columns", leave=False):
                synthetic_col = synthetic_data.iloc[:max_length, col_idx].values
                
                if len(synthetic_col) == 0:
                    continue
                
                # Apply same normalization as model.py
                synthetic_col = normalize_signal(synthetic_col)
                
                # Compute FFT of synthetic signal - now returns (frequency, magnitude)
                synthetic_freq, synthetic_mag = compute_fft(synthetic_col, fs)
                
                # Calculate all metrics on FFT domain (same as model.py)
                try:
                    pearson_val, cosine_val, kl_val, mmd_val, _, p_val = compute_metrics_fft(real_mag, synthetic_mag)
                    
                    # Update best values
                    if not np.isnan(pearson_val) and pearson_val > best_metrics['pearson']['value']:
                        best_metrics['pearson']['value'] = pearson_val
                        best_metrics['pearson']['column'] = col_idx
                        best_metrics['pearson']['data'] = synthetic_col.copy()
                    
                    if not np.isnan(cosine_val) and cosine_val > best_metrics['cosine']['value']:
                        best_metrics['cosine']['value'] = cosine_val
                        best_metrics['cosine']['column'] = col_idx
                        best_metrics['cosine']['data'] = synthetic_col.copy()
                    
                    if not np.isnan(kl_val) and kl_val < best_metrics['kl_div']['value']:
                        best_metrics['kl_div']['value'] = kl_val
                        best_metrics['kl_div']['column'] = col_idx
                        best_metrics['kl_div']['data'] = synthetic_col.copy()
                    
                    if not np.isnan(mmd_val) and mmd_val < best_metrics['mmd']['value']:
                        best_metrics['mmd']['value'] = mmd_val
                        best_metrics['mmd']['column'] = col_idx
                        best_metrics['mmd']['data'] = synthetic_col.copy()
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"    Error processing {class_name}: {str(e)}")
            continue
        
        best_results[class_name] = best_metrics
    
    return best_results

def plot_fft_subplots_multicolumn(real_base_dir, best_results, class_names,
                                  max_length=1824, fs=12000, max_freq=1500):
    """Plot FFT comparison for each class in separate subplots using best columns"""
    
    real_file_mapping = {
        'N': 'N.csv',
        '14BA': '14BA.csv', '14IR': '14IR.csv', '14OR': '14OR.csv',
        '7BA': '7BA.csv', '7IR': '7IR.csv', '7OR': '7OR.csv',
        '21BA': '21BA.csv', '21IR': '21IR.csv', '21OR': '21OR.csv', 'BA28':'BA28.csv','IR28':'IR28.csv'
    }
    
    # Create subplots: 5 rows x 2 columns for 10 classes
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20, 20))
    axs = axs.flatten()
    
    for i, class_name in enumerate(class_names):
        ax = axs[i]
        
        if class_name not in best_results:
            ax.set_title(f'{class_name} (No Data)', fontsize=12)
            continue
            
        # Load real data
        real_file = real_file_mapping.get(class_name)
        real_path = os.path.join(real_base_dir, real_file)
        
        if not os.path.exists(real_path):
            ax.set_title(f'{class_name} (No Real Data)', fontsize=12)
            continue
            
        try:
            real_data = pd.read_csv(real_path, header=None)
            real = normalize_signal(real_data.iloc[:, 0].values[:max_length])
            
            # Get best synthetic data (using best Pearson column)
            best_synthetic = best_results[class_name]['pearson']['data']
            if best_synthetic is None:
                ax.set_title(f'{class_name} (No Synthetic Data)', fontsize=12)
                continue
            
            # Compute FFTs - now correctly returns (frequency, magnitude)
            real_freqs, real_fft = compute_fft(real, fs)
            synth_freqs, synth_fft = compute_fft(best_synthetic, fs)
            
            # Apply frequency mask
            real_mask = real_freqs <= max_freq
            synth_mask = synth_freqs <= max_freq
            
            # Plot on individual subplot
            ax.plot(real_freqs[real_mask], real_fft[real_mask], 
                   label=f'{class_name} Real', linewidth=1.5, alpha=0.8, color='blue')
            ax.plot(synth_freqs[synth_mask], synth_fft[synth_mask], 
                   label=f'{class_name} Synthetic', linewidth=1.5, 
                   linestyle='--', alpha=0.8, color='red')
            
            # Formatting for each subplot
            ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Magnitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Add metrics as text
            pcc_value = best_results[class_name]['pearson']['value']
            col_idx = best_results[class_name]['pearson']['column']
            ax.text(0.02, 0.98, f'PCC: {pcc_value:.4f}\nCol: {col_idx}', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            print(f"Error plotting FFT for {class_name}: {str(e)}")
            ax.set_title(f'{class_name} (Error)', fontsize=12)
            continue
    
    # Overall figure formatting
    fig.suptitle(f'FFT Magnitude Spectrum Comparison - Best Pearson Columns (0-{max_freq} Hz)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('fft_comparison_best_columns.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_timeseries(real_base_dir, best_results, class_names, max_length=1824):
    """Plot time series comparison for all classes in separate subplots"""
    
    real_file_mapping = {
        'N': 'N.csv',
        '14BA': '14BA.csv', '14IR': '14IR.csv', '14OR': '14OR.csv',
        '7BA': '7BA.csv', '7IR': '7IR.csv', '7OR': '7OR.csv',
        '21BA': '21BA.csv', '21IR': '21IR.csv', '21OR': '21OR.csv','BA28': 'BA28.csv','IR28':'IR28.csv'
    }
    
    # Create subplots: 5 rows x 2 columns for 10 classes
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20, 25))
    axs = axs.flatten()
    
    time_axis = np.arange(max_length)
    
    for i, class_name in enumerate(class_names):
        ax = axs[i]
        
        if class_name not in best_results:
            ax.set_title(f'{class_name} (No Data)', fontsize=12)
            continue
            
        # Load real data
        real_file = real_file_mapping.get(class_name)
        real_path = os.path.join(real_base_dir, real_file)
        
        if not os.path.exists(real_path):
            ax.set_title(f'{class_name} (No Real Data)', fontsize=12)
            continue
            
        try:
            real_data = pd.read_csv(real_path, header=None)
            real_signal = normalize_signal(real_data.iloc[:, 0].values[:max_length])
            
            # Get best synthetic data (using best Pearson column)
            best_synthetic = best_results[class_name]['pearson']['data']
            if best_synthetic is None:
                ax.set_title(f'{class_name} (No Synthetic Data)', fontsize=12)
                continue
            
            # Plot time series
            ax.plot(time_axis, real_signal, 
                   label=f'{class_name} Real', linewidth=0.5, alpha=0.8, color='blue')
            ax.plot(time_axis, best_synthetic, 
                   label=f'{class_name} Synthetic', linewidth=0.5, 
                   linestyle='--', alpha=0.8, color='red')
            
            # Formatting for each subplot
            ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel('Normalized Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Add metrics as text
            pcc_value = best_results[class_name]['pearson']['value']
            col_idx = best_results[class_name]['pearson']['column']
            ax.text(0.02, 0.98, f'PCC: {pcc_value:.4f}\nCol: {col_idx}', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set reasonable y-axis limits
            ax.set_ylim([-1.1, 1.1])
            
        except Exception as e:
            print(f"Error plotting time series for {class_name}: {str(e)}")
            ax.set_title(f'{class_name} (Error)', fontsize=12)
            continue
    
    # Overall figure formatting
    fig.suptitle(f'Time Series Comparison - Best Pearson Columns ({max_length} samples)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('timeseries_comparison_best_columns.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_best_results_multicolumn(best_results):
    """Print best results for multicolumn analysis"""
    print("\n" + "="*80)
    print("BEST METRIC VALUES FOR EACH CLASS (FFT DOMAIN ANALYSIS)")
    print("="*80)
    
    for class_name, metrics in best_results.items():
        print(f"\nClass: {class_name}")
        print("-" * 60)
        print(f"Best Pearson Correlation: {metrics['pearson']['value']:.6f} (Column: {metrics['pearson']['column']})")
        print(f"Best Cosine Similarity:   {metrics['cosine']['value']:.6f} (Column: {metrics['cosine']['column']})")
        print(f"Best KL Divergence:       {metrics['kl_div']['value']:.6f} (Column: {metrics['kl_div']['column']})")
        print(f"Best MMD:                 {metrics['mmd']['value']:.6f} (Column: {metrics['mmd']['column']})")

def main():
    """Main execution function"""
    
    # Configuration - Update these paths according to your directory structure
    real_base_dir = "CWRU_data"  # Directory containing N.csv, 7BA.csv, etc.
    synthetic_base_dir = "WGAN-GP/self attention"  # Directory containing class_name.csv files with 100 columns
    
    class_names = ['N', '14BA', '14IR', '14OR', '7BA', '7IR', '7OR', '21BA', '21IR', '21OR','BA28','IR28']
    
    print("CWRU DATASET SYNTHETIC DATA QUALITY EVALUATION - FFT DOMAIN ANALYSIS")
    print("=" * 80)
    print(f"Real data directory: {real_base_dir}")
    print(f"Synthetic data directory: {synthetic_base_dir}")
    print(f"Classes to process: {', '.join(class_names)}")
    print(f"Time series length: 1824 samples")
    print(f"Evaluation domain: FFT (same as model.py)")
    print("=" * 80)
    
    # Evaluate metrics for all classes using FFT domain comparison
    best_results = evaluate_class_metrics_multicolumn(real_base_dir, synthetic_base_dir, class_names, max_length=1824)
    
    # Print results
    print_best_results_multicolumn(best_results)
    
    # Create individual FFT comparison subplots
    print(f"\n{'='*80}")
    print("GENERATING FFT COMPARISON PLOTS")
    print("="*80)
    
    plot_fft_subplots_multicolumn(real_base_dir, best_results, class_names, max_length=1824)
    
    # Create time series comparison plot
    print(f"\n{'='*80}")
    print("GENERATING TIME SERIES COMPARISON PLOTS")
    print("="*80)
    
    plot_combined_timeseries(real_base_dir, best_results, class_names, max_length=1824)
    
    print("\nEvaluation completed successfully!")
    print("FFT comparison plots saved as: fft_comparison_best_columns.png")
    print("Time series comparison plots saved as: timeseries_comparison_best_columns.png")

if __name__ == "__main__":
    main()
