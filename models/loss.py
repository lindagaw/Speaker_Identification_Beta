import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Add, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))
