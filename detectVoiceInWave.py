from vad import VoiceActivityDetector
import argparse
import json


def save_to_file(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze input wave-file and save detected speech interval to json file.')
    parser.add_argument('inputfile', metavar='INPUTWAVE',
                        help='the full path to input wave file')
    parser.add_argument('outputfile', metavar='OUTPUTFILE',
                        help='the full path to output json file to save detected speech intervals')
    parser.add_argument('outputwav', metavar='OUTPUTWAVE',
                        help='the full path to output wave file')
    args = parser.parse_args()
    
    v = VoiceActivityDetector(args.inputfile)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    v.save_output_wav(speech_labels, args.outputwav)
    save_to_file(speech_labels, args.outputfile)
