#!/usr/bin/env python
import numpy as np
from asterisk.agi import *
from scipy.io import wavfile
import os
from scikits.audiolab import Format, Sndfile
from tempfile import mkstemp
from pydub import AudioSegment
from background_processing import detect_speech

# path-related variables
PATH = "/home/hieung1707/"
PREVOICE_DIR = 'python_code/voice_detection/prerecorded_voices/'
FILE_NAME = 'sample.mp3'

# audio-related constant
SAMPLE_RATE = 8000.0
CHUNK_TIME = 0.05
BYTES_PER_SAMPLE = 2
CHUNK = int(BYTES_PER_SAMPLE * SAMPLE_RATE * CHUNK_TIME)

# time-related variables
MIN_SPOKEN_TIME = 0.2
SILENT_TIME = 1.0
TIMEOUT = 7.0

FD = 3
windows = 0
loops = 0
data = np.array([])
has_spoken = False


def rms(samples):  # Root-mean-square
    int64_samples = np.array(samples).astype(np.int64)
    return np.sqrt(np.mean(np.square(int64_samples)))


def record():
    global data, has_spoken
    # patience = 20
    # n_silence = 0
    agi.verbose('start recording')
    # start_recording = True
    # --- new variables --- #
    cons_sil_time = 0.0
    cons_speech_time = 0.0
    total_time = 0.0
    # --------------------- #

    # --- old logic --- #
    # while start_recording:
    #     raw_samples = file_descriptor.read(CHUNK)
    #     new_samples = np.fromstring(raw_samples, dtype=np.int16)
    #     data = np.append(data, new_samples)
    #     v.data = data
    # ---------------- #
    
    # --- new logic --- #

    while cons_sil_time < SILENT_TIME:
        raw_samples = file_descriptor.read(CHUNK)
        new_samples = np.fromstring(raw_samples, dtype=np.int16)
        # new_samples = np.fromstring(raw_samples, dtype=np.int16)
        total_time += CHUNK_TIME
        spoke = detect_speech(new_samples)
        # agi.verbose(spoke)
        if spoke:
            cons_sil_time = 0.0
            cons_speech_time += CHUNK_TIME
            if cons_speech_time >= MIN_SPOKEN_TIME and not has_spoken:
                agi.verbose('has spoken')
                has_spoken = True
                data = np.append(data[data.size - int(0.5*SAMPLE_RATE):] if data.size - 0.5*SAMPLE_RATE > 0 else data, new_samples)
        else:
            cons_speech_time = 0.0
            if has_spoken:
                cons_sil_time += CHUNK_TIME
        data = np.append(data, new_samples)
        if total_time >= TIMEOUT:
            break

    # ---------------- #
    n_samples = len(data)
    padding = np.zeros(int(0.3 * SAMPLE_RATE),np.int16)
    data = np.append(padding, data)
    data = np.append(data, padding)

    n_secs = n_samples / SAMPLE_RATE
    agi.verbose("collected " + str(n_samples) + " for " + str(n_secs))


def save_record():
    global data
    n_channels, fmt = 1, Format('flac', 'pcm16')
    caller_id = agi.get_variable("CALLERID(num)")
    file_name = 'TmpSpeechFile_' + caller_id + '.flac'
    _, temp_sound_file = mkstemp(file_name)
    flac_file = Sndfile(temp_sound_file, 'w', fmt, n_channels, SAMPLE_RATE)

    flac_file.write_frames(data)
    flac_audio = AudioSegment.from_file(temp_sound_file, "flac")
    flac_audio.export(PATH + FILE_NAME, format="mp3")


if __name__ == "__main__":
    agi = AGI()
    fs, wav_file = wavfile.read(PATH + PREVOICE_DIR + "tongdai.wav")
    TIMEOUT = wav_file.size / fs
    file_descriptor = os.fdopen(FD, 'r')
    record()
    save_record()
    # reduce_noise()
    agi.set_variable("has_spoken", 1 if has_spoken else 0)
    agi.verbose('finish')