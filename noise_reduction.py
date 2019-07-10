from scipy.signal import stft, istft
from scipy.io import wavfile
import numpy as np
import math


def reduce_noise():
    fs, signal = wavfile.read('/home/hieung1707/sample.wav')
    signal = np.array(signal)
    T = signal.size / fs
    t = np.linspace(0, T, signal.size, endpoint=False)
    freqs, times, signal_stft = stft(signal, fs, 'hamming', nperseg=200, noverlap=100, nfft=None, padded=True)

    # calculate magnitude spctrogram
    mag_spec = np.abs(signal_stft)

    # calculate phase spectrogram
    phase_spec = []
    for window in signal_stft:
        window_phase = np.array([])
        for pt in window:
            phase = math.atan(pt.imag / pt.real)
            window_phase = np.append(window_phase, phase)
        phase_spec.append(window_phase)
    phase_spec = np.array(phase_spec)

    # denoise
    noise_est = np.mean(mag_spec[:3])
    new_mag_spec = []
    for window in mag_spec:
        new_window = []
        for pt in window:
            new_pt = pt - noise_est if pt > noise_est else 0
            new_window.append(new_pt)
        new_mag_spec.append(new_window)
    new_mag_spec = np.array(new_mag_spec)

    new_signal_stft = new_mag_spec * np.exp(phase_spec * 1j)
    times, new_signal = istft(new_signal_stft, fs, 'hamming', nperseg=200, noverlap=100, nfft=None)
    signal = np.append(signal, np.zeros((new_signal.size - signal.size)))
    signal[new_signal == 0] = 0
    signal = signal * 1.0 / max(signal)
    wavfile.write('/home/hieung1707/sample.wav', fs, signal)