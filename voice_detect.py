#!/usr/bin/env python
import os
import threading
import wave
import pyaudio
import numpy as np
from asterisk.agi import *
from scikits.audiolab import Format, Sndfile
from tempfile import mkstemp
from pydub import AudioSegment
from vad import VoiceActivityDetector
from stt_api_connector import stt

# path-related variables
PATH = "/home/hieung1707/"
PREVOICE_DIR = 'python_code/voice_detection/prerecorded_voices/'
FILE_NAME = 'sample.wav'

# audio-related constant
SAMPLE_RATE = 8000.0
CHUNK_TIME = 0.2
BYTES_PER_SAMPLE = 2
CHUNK = int(BYTES_PER_SAMPLE * SAMPLE_RATE * CHUNK_TIME)

# time-related variables
TIMEOUT = 20.0
SILENT_TIME = 1
MIN_SPOKEN_TIME = 0.2
FD = 3
windows = 0
loops = 0
data = np.array([])
start_recording = False


def rms(samples):  # Root-mean-square
    int64_samples = np.array(samples).astype(np.int64)
    return np.sqrt(np.mean(np.square(int64_samples)))


def record():
    global data, start_recording
    CHUNK = int(BYTES_PER_SAMPLE * SAMPLE_RATE * CHUNK_TIME)
    # patience = 20
    # n_silence = 0
    agi.verbose('start recording')
    start_recording = True
    while start_recording:
        raw_samples = file_descriptor.read(CHUNK)
        new_samples = np.fromstring(raw_samples, dtype=np.int16)
        data = np.append(data, new_samples)
        v.data = data

    n_samples = len(data)
    n_secs = n_samples / SAMPLE_RATE
    agi.verbose("collected " + str(n_samples) + " for " + str(n_secs))
    return data


def background_process():
    global start_recording
    sil_time = 0.0
    last_pos = v.sample_start
    consecutive_speech_time = 0.0
    has_spoken = False
    while sil_time < SILENT_TIME:
        if not start_recording:
            continue
        raw_detection = v.detect_speech()
        v.detect_speech_in_windows(raw_detection)
        if v.sample_start == last_pos:
            continue
        last_pos = v.sample_start
        current_record_time = v.data.size / v.rate
        if v.is_speech == 0:
            consecutive_speech_time = 0.0
            if has_spoken:
                sil_time += v.sample_window if sil_time == 0 else v.sample_overlap
                agi.verbose("silent time: " + str(sil_time))
        else:
            consecutive_speech_time += v.sample_window if sil_time == 0 else v.sample_overlap
            if consecutive_speech_time >= MIN_SPOKEN_TIME:
                has_spoken = True
            sil_time = 0.0
        if current_record_time >= TIMEOUT:
            break
        # agi.verbose('processing, silent time: ' + str(sil_time))
    start_recording = False


def save_record():
    global data
    n_channels, fmt = 1, Format('flac', 'pcm16')
    caller_id = agi.get_variable("CALLERID(num)")
    agi.verbose('caller id is ' + caller_id)
    file_name = 'TmpSpeechFile_' + caller_id + '.flac'
    _, temp_sound_file = mkstemp(file_name)
    flac_file = Sndfile(temp_sound_file, 'w', fmt, n_channels, SAMPLE_RATE)

    flac_file.write_frames(data)
    flac_audio = AudioSegment.from_file(temp_sound_file, "flac")
    flac_audio.export("/home/hieung1707/sample.mp3", format="mp3")


if __name__ == "__main__":
    agi = AGI()
    detect_sound = False
    agi.verbose("init success")
    file_descriptor = os.fdopen(FD, 'r')
    v = VoiceActivityDetector(data)
    file_path = '//home/hieung1707/python_code/voice_detection/prerecorded_voices/tongdai.wav'
    thread1 = threading.Thread(target=background_process)
    thread1.start()
    data = record()
    save_record()
    labels = v.convert_windows_to_readible_labels(v.windows)
    v.save_output_wav(labels, PATH + 'output.wav')
    text, intent = stt()
    agi.verbose(text)
    agi.verbose(intent)
    agi.verbose("finish")