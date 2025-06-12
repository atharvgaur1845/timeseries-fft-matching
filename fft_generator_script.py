import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("gpt-4.1_data.csv", header=None)
signal = data[0].values
#sampling freq
fs = 1000 
#time vector
t = np.arange(len(signal)) / fs

#main fft via nupmpy
N = len(signal)
fft_vals = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, d=1/fs)
#only positive part
positive_freqs = fft_freqs[:N // 2]
fft_magnitude = np.abs(fft_vals[:N // 2]) * 2 / N  


plt.figure(figsize=(16, 9))
plt.subplot(2, 1, 1)
plt.plot(t[:1000], signal[:1000]) 
plt.title("Time-Domain Signal of GPT-4.1 Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Frequency-Domain Plot
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, fft_magnitude)
plt.title("FFT of GPT-4.1 Data")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
