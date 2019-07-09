#!/usr/bin/env python
import numpy as np
from asterisk.agi import *
from stt_api_connector import stt
from scikits.audiolab import Format, Sndfile
from tempfile import mkstemp
import os


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
TIMEOUT = 5.0
SILENT_TIME = 2.0
MIN_SPOKEN_TIME = 0.2
FD = 3
windows = 0
loops = 0
data = np.array([])


def rms(samples):  # Root-mean-square
    int64_samples = np.array(samples).astype(np.int64)
    return np.sqrt(np.mean(np.square(int64_samples)))


def record():
    global data
    # patience = 20
    # n_silence = 0
    agi.verbose('start recording')
    # start_recording = True
    # --- new variables --- #
    cons_sil_time = 0.0
    cons_speech_time = 0.0
    has_spoken = False
    # --------------------- #

    # --- old logic --- #
    # while start_recording:
    #     raw_samples = file_descriptor.read(CHUNK)
    #     new_samples = np.fromstring(raw_samples, dtype=np.int16)
    #     data = np.append(data, new_samples)
    #     v.data = data
    # ---------------- #
    
    # --- new logic --- #
    avg_power = 0.0
    counter = 0

    while cons_sil_time < SILENT_TIME:
        raw_samples = file_descriptor.read(CHUNK)
        new_samples = np.fromstring(raw_samples, dtype=np.int16)
        power = rms(new_samples)
        if counter < 3:
            avg_power += power
        elif counter == 3:
            avg_power /= 3
        counter += 1
        # agi.verbose('muc nang luong: ' + str(power))
        if power / avg_power >= 1.5:
            cons_sil_time = 0.0
            cons_speech_time += CHUNK_TIME
            if cons_speech_time >= MIN_SPOKEN_TIME:
                has_spoken = True
        else:
            cons_speech_time = 0.0
            if has_spoken:
                cons_sil_time += CHUNK_TIME
                # agi.verbose('silent time: ' + str(cons_sil_time))
        data = np.append(data, new_samples)

    # ---------------- #
    n_samples = len(data)
    n_secs = n_samples / SAMPLE_RATE
    agi.verbose("collected " + str(n_samples) + " for " + str(n_secs))


# def background_process():
#     global start_recording
#     sil_time = 0.0
#     last_pos = v.sample_start
#     cons_speech_time = 0.0
#     has_spoken = False
#     while sil_time < SILENT_TIME:
#         if not start_recording:
#             continue
#         raw_detection = v.detect_speech()
#         v.detect_speech_in_windows(raw_detection)
#         if v.sample_start == last_pos:
#             continue
#         last_pos = v.sample_start
#         current_record_time = v.data.size / v.rate
#         if v.is_speech == 0:
#             cons_speech_time = 0.0
#             if has_spoken:
#                 sil_time += v.sample_window if sil_time == 0 else v.sample_overlap
#                 agi.verbose("silent time: " + str(sil_time))
#         else:
#             cons_speech_time += v.sample_window if sil_time == 0 else v.sample_overlap
#             if cons_speech_time >= MIN_SPOKEN_TIME:
#                 has_spoken = True
#             sil_time = 0.0
#         if current_record_time >= TIMEOUT:
#             break
#         # agi.verbose('processing, silent time: ' + str(sil_time))
#     start_recording = False


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
    file_descriptor = os.fdopen(FD, 'r')
# # v = VoiceActivityDetector(data)
# # thread1 = threading.Thread(target=background_process)
# # thread1.start()
# sound_loader = SoundLoaderThread(agi)
# sound_loader.playAudio(file_path)
    record()
    save_record()
# sound_loader.stop()
# # labels = v.convert_windows_to_readible_labels(v.windows)
# # v.save_output_wav(labels, PATH + 'output.wav')
    text, intent = stt()
    agi.verbose(text)
    agi.verbose(intent)
    agi.verbose("finish")
