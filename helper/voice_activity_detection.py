from func_timeout import func_timeout, FunctionTimedOut

import os

import soundfile as sf
import shutil
# Import the voice activity detection module
import speech_recognition as sr

# disable warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import urllib3


def detect_voice_activity(wav_path):
    r = sr.Recognizer()

    with sr.AudioFile(wav_path) as source:
        try:
            audio = func_timeout(10, r.record, args=(source, 7.884e+6))
            #audio = r.record(source, duration=7.884e+6)
        except Exception as e:
            print(e)

    try:
        transcription = func_timeout(5.5, r.recognize_google, args=[audio])
        #print('{} is transcribed as {}.'.format(wav_path, transcription))
        return True
    except Exception as e:
        #print('{} is not speech.'.format(wav_path))
        os.remove(wav_path)
        return False
