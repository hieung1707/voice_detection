import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import  spectrogram
from scipy.fftpack import fft, fftfreq
import scipy

T = 1
f = 10
nsamples = 2000
t = np.linspace(0, T, nsamples, endpoint=False)
A = 5

signal = A*np.sin(2*np.pi*f*t)
signal_fft = fft(signal)
data_ampt = abs(signal_fft)
data_freq = fftfreq(nsamples, t[1] - t[0])
freqs, times, Sx = spectrogram(signal, f, nperseg=20, noverlap=10)

ampt_freq = {}
for (i, freq) in enumerate(data_freq[1:]):
    if abs(freq) not in data_ampt:
        ampt_freq[abs(freq)] = data_ampt[i] * 2
# for freq in ampt_freq:
#     print ampt_freq[freq]
ampt_freq_final = [ampt_freq[freq] for freq in ampt_freq]
print ampt_freq_final
plt.plot(ampt_freq.keys(), ampt_freq_final, label='fft')
plt.show()
# freqs *= 20
# # f, ax = plt.subplots(figsize=(4.8, 2.4))
# # ax.pcolormesh(times / times.size, freqs, 10 * np.log10(Sx), cmap='viridis')
# # ax.set_ylabel('Frequency [Hz]')
# # ax.set_xlabel('Time [s]')
# # plt.show()