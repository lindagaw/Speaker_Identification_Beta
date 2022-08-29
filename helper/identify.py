import tensorflow.keras as keras
import tensorflow as tf

import numpy as np
import pickle
import os

import sys

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))

model = keras.models.load_model('generated_models//cnn.hdf5', custom_objects={"mil_squared_error": mil_squared_error}, compile=False)

def identify_speaker(vec, threshold):
    raw_pred = np.squeeze(model.predict(vec))
    sid = np.argmax(raw_pred)

    print(raw_pred)

    if raw_pred[sid] > threshold:
        if sid == 0:
            return 'caregiver'
        else:
            return 'patient'
    else:
        return 'unregistered'
