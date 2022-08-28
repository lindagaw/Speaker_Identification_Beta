import os
import shutil
import pyaudio
from helper import record_single_window, detect_voice_activity

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

if __name__ == '__main__':

    dir = 'generated_data//'

    try:
        shutil.rmtree(dir)
    except Exception as e:
        pass

    os.makedirs(dir)


    if not os.path.isdir(dir):
        os.makedirs(dir)

    for i in range(0, 5):
        path_wav = record_single_window(CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS, dir)
        is_voice_activity = detect_voice_activity(path_wav)

        print(is_voice_activity)
