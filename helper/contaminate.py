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

def change_amplitude(emotionfile, d1, newSoundFile, d2):

    if d1 <= d2:
        sound = AudioSegment.from_file(emotionfile) - np.random.randint(0, (6 * d2/d1 - 1))
        sound.export(newSoundFile, format='wav')  ### save the new generated file in a folder
    else:
        print('Invalid distance parameters. d1 should be <= d2.')

def change_amplitude_range(emotionfile, newSoundFile, threshold):
    #amount = np.random.randint(0, threshold)
    amount = random.uniform(0, threshold)
    #print('Deamplify ' + str(emotionfile) + ' by ' + str(amount) + ' db.')
    sound = AudioSegment.from_file(emotionfile) - amount
    sound.export(newSoundFile, format='wav')  ### save the new generated file in a folder
    return amount

def deamplify_per_folder(directory):
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            soundFile = directory + file
            newSoundFile = directory + 'deamplified_' + file
            change_amplitude_range(soundFile, newSoundFile, 12)

def add_noise_and_deamplify_per_folder(directory, extension, noise_directory):
    for file in os.listdir(directory):
        if file.endswith(extension) and not file[1] == '_':
            for i in range(0, 2):
                soundFile = directory + file
                amount = change_amplitude_range(soundFile, soundFile, 1.5)
                noise = random.choice(os.listdir(noise_directory))
                random_noise = noise_directory + noise
                newSoundFile = directory + 'deamp_' + str(amount) + '_noise_' + noise[:len(noise)-5] + '_' + file
                add_noise_per_file(soundFile, random_noise, newSoundFile)
                print(newSoundFile)

def add_noise_per_file(emotionfile, bgnoise, newSoundFile):

    emotionsound = AudioSegment.from_file(emotionfile, format="wav")
    emotion_duration = emotionsound.duration_seconds * 1000
    noise = AudioSegment.from_file(bgnoise, format="wav")
    noise_duration = noise.duration_seconds * 1000

    threshold = noise_duration - emotion_duration

    if threshold > 0:
        overlay_start = np.random.randint(0, threshold)
    else:
        overlay_start = 0

    targeted_chunk = noise[overlay_start:overlay_start + emotion_duration]
    newSound = emotionsound.overlay(targeted_chunk, position=0)
    newSound=newSound[0:5000]
    newSound.export(newSoundFile, format='wav')  ### save the new generated file in a folder
