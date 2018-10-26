"""We are using the SAVEE dataset, which consists of utterances of actors speaking 15 sentences for 7 different emotions. We are only training for 'positive' and 'negative' emotions, so we need to relabel the data and convert it to spectrograms."""

from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
import librosa.output

from os import listdir
from os.path import isfile, isdir, join, basename

import argparse
import random
import re

SAVEE_EMOTION_FULL_NAME = {'a': 'anger', 'd': 'disgust', 'f': 'fear', 'h': 'happiness', 'n': 'neutral', 'sa': 'sadness', 'su': 'surprise'}

# Negative emotions are given a 0 and positive emotions are given a 1.
POS_NEG_EMOTION_MAPPING = {'a': 0, 'd': 0, 'f': 0, 'h': 1, 'n': 1, 'sa': 0, 'su': 1}

def partition_data(path, training_percentage=0.10):
    all_input_files = []
    for d in listdir(path):
        subdir_path = join(path, d)
        if isdir(subdir_path):
            all_input_files += [join(subdir_path, f) for f in listdir(subdir_path) if isfile(join(subdir_path, f))]

    random.shuffle(all_input_files)
    training_data = all_input_files[:int(np.ceil(training_percentage * len(all_input_files)))]
    testing_data = all_input_files[int(np.ceil(training_percentage * len(all_input_files))):]

    return training_data, testing_data

def sample_info(filename):
    """
    Returns the emotion and sentence number of a given file from the SAVEE dataset.
    """
    filename = filename[:-4]
    emotion_label = re.split("[^a-z]+", filename)[0]
    sentence_label = filename[len(emotion_label):]
    return emotion_label, sentence_label

def select_clip(samples, sr=16000, length_s=2):
    length_samples = int(np.floor(sr * length_s))
    return samples[int((len(samples) - length_samples) / 2) : int((len(samples) + length_samples) / 2)]

def add_background_noise(sample, path_to_chunked_noise, loudness_scaling = 0.5):
    noise_file = random.choice(listdir(path_to_chunked_noise))
    noise, rate = librosa.load(join(path_to_chunked_noise, noise_file), sr=16000, res_type='scipy')
    if len(sample) == len(noise):
        return sample + noise * loudness_scaling
    else:
        return sample + np.pad(noise * loudness_scaling, (0, len(sample) - len(noise)), 'wrap')

def data_preprocess(samples, path_to_chunked_noise, length_s=2.0, sr=16000, loudness_scaling=0.2):
    # TODO: look into normalizing data.
    shortened_clip = select_clip(samples, sr=sr)
    return add_background_noise(shortened_clip, path_to_chunked_noise, loudness_scaling=loudness_scaling)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS = None
    parser.add_argument('--audio_path', type=str, default='', help="""The
            absolute path to the directory containing the audio examples. Use
            $HOME/catkin_ws/src/robot_learning/AudioData""")
    FLAGS, _ = parser.parse_known_args()

    rate, data = wavfile.read('/home/ariana/catkin_ws/src/robot_learning/AudioData/DC/a01.wav')
    noised = data_preprocess(data, '/home/ariana/catkin_ws/src/robot_learning/BackgroundNoise/chunked')
    wavfile.write('/tmp/noised.wav', rate, noised)
