import numpy as np
import pandas as pd
import json
from scipy.fft import fft
#time domain features
def extract_time_domain_features(x):
    N = len(x)
    p1 = np.mean(x)
    p2 = np.std(x, ddof=1)
    p3= (np.mean(np.sqrt(np.abs(x))))**2
    p4 = np.mean(np.abs(x))
    p5 =np.max(np.abs(x))
    p6 =np.mean((x - p1)**3)
    p7 = np.mean((x - p1)**4)
    p8= np.mean((x - p1)**2)
    p9 = p7 / (p6**2) if p6 != 0 else 0
    p10 =p5 / p2 if p2 != 0 else 0
    p11 = p2 / p4 if p4 != 0 else 0
    p12 = p5 / p4 if p4 != 0 else 0
    return {
        'mean_value': p1,
        'standard_deviation': p2,
        'square_root_amplitude': p3,
        'absolute_mean_value': p4,
        'peak_value': p5,
        'skewness': p6,
        'kurtosis': p7,
        'variance': p8,
        'kurtosis_index': p9,
        'peak_index': p10,
        'waveform_index': p11,
        'pulse_index': p12
    }

#frequency domain features
def extract_frequency_domain_features(x):
    N = len(x)
    s = np.abs(fft(x))[:N//2] #/2 to get only real frequencies
    K= len(s)
    f = np.arange(K)  
    p13= np.mean(s)
    p14= np.std(s, ddof=1)
    p15= np.sum((s - p13)**3) / (K * (p14**3)) if p14 != 0 else 0
    p16 = np.sum((s - p13)**4) / (K * (p14**4)) if p14 != 0 else 0
    p17 =np.sum(f * s) / np.sum(s) if np.sum(s) != 0 else 0
    p18 = np.sqrt(np.sum(((f - p17)**2) * s) / np.sum(s)) if np.sum(s) != 0 else 0
    p19 =np.sqrt(np.sum((f**2) * s) / np.sum(s)) if np.sum(s) != 0 else 0
    p20 = np.sum((f**4) * s) / np.sum((f**2) * s) if np.sum((f**2) * s) != 0 else 0
    p21= np.sum((f**2) * s) / np.sum(s) if np.sum(s) != 0 else 0
    p22= p18 / p17 if p17 != 0 else 0
    p23 = np.sum(((f - p17)**3) * s) / (K * (p18**3)) if p18 != 0 else 0
    p24 = np.sum(((f - p17)**4) * s) / (K * (p18**4)) if p18 != 0 else 0
    return {
        'frequency_mean_value': p13,
        'frequency_variance': p14,
        'frequency_skewness': p15,
        'frequency_kurtosis': p16,
        'gravity_frequency': p17,
        'frequency_standard_deviation': p18,
        'frequency_root_mean_square': p19,
        'average_frequency': p20,
        'regularity_degree': p21,
        'variation_parameter': p22,
        'eighth_order_moment': p23,
        'sixteenth_order_moment': p24
    }

def extract_features_from_csv(csv_file_path, json_file_path):
    df = pd.read_csv(csv_file_path)
    features_list = []
    for column in df.columns:
        x = df[column].values
        time_features = extract_time_domain_features(x)
        freq_features = extract_frequency_domain_features(x)
        features = {**time_features, **freq_features}
        features_list.append({"sensor": column, "features": features})
    with open(json_file_path, 'w') as f:
        json.dump(features_list, f, indent=4)

#calling the function
extract_features_from_csv('original_data.csv', 'features.json')
