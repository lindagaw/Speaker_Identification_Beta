import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Add, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend
from sklearn.metrics import accuracy_score

adam = tf.keras.optimizers.Adam(learning_rate=1e-5)

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))

def train_cnn(X_train, y_train, X_test, y_test, X_val, y_val):

    model = keras.Sequential()
    model.add(Convolution1D(filters= 1500, kernel_size=2, strides=2, activation='relu', input_shape=(48, 272) ))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Convolution1D(filters= 500, kernel_size=2, strides=2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Convolution1D(filters= 500, kernel_size=2, strides=2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Convolution1D(filters= 500, kernel_size=2, strides=2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Flatten())

    for i in range(0, 1):
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss=['categorical_crossentropy'], optimizer=adam, metrics=['accuracy', mil_squared_error])

    print("Fit model on training data")
    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=150,
        validation_data=(X_val, y_val), verbose=1
    )

    #from sklearn.metrics import f1_score

    y_preds = [np.argmax(val) for val in model.predict(X_test)]
    y_trues = [np.argmax(val) for val in y_test]
    print(accuracy_score(y_trues, y_preds))

    model.save('generated_models//cnn.hdf5')
    return model
