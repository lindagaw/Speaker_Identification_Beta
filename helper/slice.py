from pydub import AudioSegment
import time
import os
import numpy as np
import pickle
import random
from librosa import load
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def slice_audios(path, dest):
    for old_audio in os.listdir(dest):
        os.remove(dest + old_audio)

    for audio_index in range(0, len(os.listdir(path))):
        target_audio_path =  path + os.listdir(path)[audio_index]
        print('input audio: ' + target_audio_path)

        target_audio = AudioSegment.from_wav(target_audio_path)
        target_duration = target_audio.duration_seconds

        folds = int(target_duration/5.0)
        for fold in range(0, folds+1):

            start_time = time.time()

            start_time = fold * 5000  # Works in milliseconds
            end_time = (fold + 1) * 5000
            new_audio = target_audio[start_time:end_time]

            components = target_audio_path.split('/')
            filename = components[len(components)-1]

            new_audio_path = dest + filename[:len(filename)-4] + '_' + str(fold) + '.wav'
            print('generated the slice of the audio segment at index ' + str(fold) )
            new_audio.export(new_audio_path, format="wav")
