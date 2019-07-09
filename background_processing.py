import numpy as np

sample_window_time = 0.05
sample_overlap_time = 0.025
speech_window = 0.5
speech_energy_threshold = 0.7
speech_start_band = 50
speech_end_band = 8000
rate = 8000.0


def _calculate_frequencies(audio_data):
    data_freq = np.fft.fftfreq(len(audio_data), 1.0 / rate)
    data_freq = data_freq[1:]
    return data_freq


def _calculate_amplitude(audio_data):
    data_ampl = np.abs(np.fft.fft(audio_data))
    data_ampl = data_ampl[1:]
    return data_ampl


def _calculate_energy(data):
    data_amplitude = _calculate_amplitude(data)
    data_energy = data_amplitude ** 2
    return data_energy


def _connect_energy_with_frequencies(data_freq, data_energy):
    energy_freq = {}
    for (i, freq) in enumerate(data_freq):
        if abs(freq) not in energy_freq:
            energy_freq[abs(freq)] = data_energy[i] * 2
    return energy_freq


def _calculate_normalized_energy(data):
    data_freq = _calculate_frequencies(data)
    data_energy = _calculate_energy(data)
    # data_energy = _znormalize_energy(data_energy) #znorm brings worse results
    energy_freq = _connect_energy_with_frequencies(data_freq, data_energy)
    return energy_freq


def _sum_energy_in_band(energy_frequencies, start_band, end_band):
    sum_energy = 0
    for f in energy_frequencies.keys():
        if start_band < f < end_band:
            sum_energy += energy_frequencies[f]
    return sum_energy


def _median_filter(x, k):
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def _smooth_speech_detection(detected_windows):
    median_window = int(speech_window / sample_window_time)
    if median_window % 2 == 0: median_window = median_window - 1
    median_energy = _median_filter(detected_windows[:, 1], median_window)
    return median_energy


def detect_speech(data):
    """ Detects speech regions based on ratio between speech band energy
    and total energy.
    Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).
    """
    detected_windows = np.array([])
    sample_window = int(rate * sample_window_time)
    sample_start = 0
    sample_end = sample_start + sample_window if sample_start + sample_window <= len(data) else len(data)
    start_band = speech_start_band
    end_band = speech_end_band
    data_window = data[sample_start:sample_end]
    energy_freq = _calculate_normalized_energy(data_window)
    sum_voice_energy = _sum_energy_in_band(energy_freq, start_band, end_band)
    sum_full_energy = sum(energy_freq.values())
    speech_ratio = sum_voice_energy / sum_full_energy
    # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
    speech_ratio = speech_ratio > speech_energy_threshold
    detected_windows = np.append(detected_windows, [sample_start, speech_ratio])
    detected_windows = detected_windows.reshape(int(len(detected_windows) / 2), 2)  # type: None
    detected_windows[:, 1] = _smooth_speech_detection(detected_windows)
    return speech_ratio, detected_windows
