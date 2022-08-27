from helper import slice_audios, delete_dir
from helper import extract_features_for_all_wavs, add_noise_and_deamplify_per_folder
from models import train_cnn, mil_squared_error, get_optimizer

import os
import numpy as np
import pretty_errors
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path_caregiver = 'singles//1-caregiver//'
dest_caregiver = 'singles//1-caregiver-sliced//'

path_patient = 'singles//2-patient//'
dest_patient = 'singles//2-patient-sliced//'

noise_directory = 'noise_home//'

delete_dir(dest_caregiver)
delete_dir(dest_patient)

os.makedirs(dest_caregiver)
os.makedirs(dest_patient)



def start_SID_train():


    slice_audios(path_caregiver, dest_caregiver)
    slice_audios(path_patient, dest_patient)


    add_noise_and_deamplify_per_folder(dest_caregiver, '.wav', noise_directory)
    add_noise_and_deamplify_per_folder(dest_patient, '.wav', noise_directory)

    
    X_caregiver, y_caregiver = extract_features_for_all_wavs(dest_caregiver, 0)
    X_patient, y_patient = extract_features_for_all_wavs(dest_patient, 1)


    X = np.vstack((X_caregiver, X_patient))
    y = to_categorical( np.vstack((y_caregiver, y_patient)) )

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    model = train_cnn(X_train, y_train, X_test, y_test, X_val, y_val)
    

if __name__ == "__main__":
    start_SID_train()
