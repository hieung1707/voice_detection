from scipy.io import wavfile
import numpy as np
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add silent at the end of file.')
    parser.add_argument('inputname', metavar='INPUTWAVE',
                            help='name of input wave file')
    args = parser.parse_args()
    file_name = args.inputname

    wav_temp_file_name = file_name + '_temp.wav'
    wav_file_name = file_name + '.wav'
    gsm_file_name = file_name + '.gsm'
    silent = 5
    fs, data = wavfile.read(wav_file_name)
    data = np.append(data, np.zeros(fs * silent, np.int16))
    wavfile.write(wav_temp_file_name, fs, data)
    cmd = 'sox ' + wav_temp_file_name + ' -r 8000 -c 1 ' + gsm_file_name
    os.system(cmd)
    os.remove(wav_temp_file_name)
