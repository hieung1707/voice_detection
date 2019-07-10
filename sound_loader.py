from scipy.io import wavfile
import numpy as np

fs = 8000.0
T = 5.0
nsamples = int(T *fs)
a = np.zeros(nsamples, np.int16)
wavfile.write('waiting.wav', int(fs), a)