import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Add, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend

def get_optimizer():
    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
