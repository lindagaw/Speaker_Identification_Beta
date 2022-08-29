import os
import shutil
import pyaudio
import numpy as np
from helper import record_single_window, detect_voice_activity, identify_speaker
from helper import extract_feats_single_wav

import sys
import pretty_errors

sys.path.insert(1, 'helper//')

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
        if detect_voice_activity(path_wav):
            vec = np.expand_dims(extract_feats_single_wav(path_wav), axis=0)
            sid = identify_speaker(vec, 0.8)

            print(sid)