import os
import shutil
import pyaudio
import numpy as np
from helper import record_single_window, detect_voice_activity, identify_speaker
from helper import extract_feats_single_wav

import sys
import os
import pretty_errors


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5


if __name__ == '__main__':

    dir = 'generated_data//'

    '''
    try:
        shutil.rmtree(dir)
    except Exception as e:
        pass
    os.makedirs(dir)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    '''
    total = 100
    for i in range(0, total):
        print('Recording the {}th of the {} wavs...'.format(i, total))
        path_wav = record_single_window(CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS, dir)
        if detect_voice_activity(path_wav):
            vec = np.expand_dims(extract_feats_single_wav(path_wav), axis=0)
            sid = identify_speaker(vec, 0.8)

            filename = path_wav.split('//')[len(path_wav.split('//'))-1]
            where = sys.argv[1]
            background = sys.argv[2]
            new_path_wav = os.path.join(dir, where + '_' + background + '_classified-as_' + str(sid) + '_' + filename)
            os.rename(path_wav, new_path_wav)

            print(new_path_wav)
