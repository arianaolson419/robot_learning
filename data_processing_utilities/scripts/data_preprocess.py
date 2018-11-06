"""This module contains functions to be used to preprocess audio data before
turning it into a spectrogram for input into the neural network and functions
to partition and label spectrograms for input into the network.

Authors: Anna Buchele, Ariana Olson
"""

import numpy as np
import librosa

from os import listdir
from os.path import isfile, isdir, join

import random
import re

# For nicer print statements
SAVEE_EMOTION_FULL_NAME = {'a': 'anger', 'd': 'disgust', 'f': 'fear', 'h': 'happiness', 'n': 'neutral', 'sa': 'sadness', 'su': 'surprise'}

# Negative emotions are given a 0 and positive emotions are given a 1.
POS_NEG_EMOTION_MAPPING = {'a': 0, 'd': 0, 'f': 0, 'h': 1, 'n': 1, 'sa': 0, 'su': 1}

def partition_data(path, training_percentage=0.90):
    """Partitions all generated spectrograms into a training and a validation set.

    Files are randomly chosen to be put in one set or the other, and files are
    unique to one set (no file is included in both trainig and testing.

    Parameters
    ----------
    path: the path to the directory containing all spectrograms.
    training_percentage: the percent of the data to use for the training set.

    Returns
    -------
    a tuple containing a list of file paths in the training set and a list of
    file paths in the validation set.
    """
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

    Parameters
    ----------
    filename: the name of a spectrogram file of the form <emotion_label><sentence_number>.npy

    Returns
    -------
    a tuple containing the emotion label and e=sentence number.
    """
    filename = filename[:-4]
    emotion_label = re.split("[^a-z]+", filename)[0]
    sentence_label = filename[len(emotion_label):]
    return emotion_label, sentence_label

def get_label(filename):
    """
    Determines the emotion of the sample and formats it into a label to feed into the network.

    Parameters
    ----------

    Returns
    -------
    a tuple containing a 1.0 and a 0.0 in either order (either (1.0, 0.0) or
    (0.0, 1.0)). A one in postion 0 represents a negative label and a one in
    position 1 represents a positive label.
    """
    filename = filename.split('/')[-1]
    e, _ = sample_info(filename)
    label = np.zeros((2,))
    label[POS_NEG_EMOTION_MAPPING[e]] = 1.0
    return label

def select_clip(samples, sr=16000, length_s=2):
    """Selects a portion of the audio clip of a specified length.

    Parameters
    ----------
    samples: a numpy array of single-channel floating point audio data.
    sr: the samples rate of the audio in Hz. Defaults to 16000Hz.
    length_s: the length of the clip to select in seconds. Defaults to 2s.

    Returns
    -------
    a numpy array of single-channel floating point audio at the specified sample
        rate and duration.
    """
    length_samples = int(np.floor(sr * length_s))
    if len(samples) < length_samples:
        return np.pad(samples, (0, length_samples - len(samples)), 'wrap')
    return samples[int((len(samples) - length_samples) / 2) : int((len(samples) + length_samples) / 2)]

def add_background_noise(sample, path_to_chunked_noise, loudness_scaling = 0.5):
    """Adds a random clip of background noise to an audio clip.

    Parameters
    ----------
    sample: a shortened clip of audio from the dataset. NOTE: the background
        noise is chunked such that all clips are 2s long.
    path_to_chunked_noise: the path to the directory containing 2s clips of background noise.
    loudness_scaling: the factor to scale:wa
    """
    noise_file = random.choice(listdir(path_to_chunked_noise))
    noise, rate = librosa.load(join(path_to_chunked_noise, noise_file), sr=16000, res_type='scipy')
    if len(sample) == len(noise):
        return sample * (1 - loudness_scaling) + noise * loudness_scaling
    elif len(sample) > len(noise):
        return sample * (1 - loudness_scaling) + np.pad(noise * loudness_scaling, (0, len(sample) - len(noise)), 'wrap')
    else:
        return sample * (1 - loudness_scaling) + noise[0 : len(sample)] * loudness_scaling

def normalize_audio(data):
    """Calculates and normalizes the loudness of the audio to a target value.

    Parameters
    ----------
    data: a numpy array of flaoting point single-channel audio data to be normalized.

    Returns
    -------
    a numpy array of floating point single-channel audio data.
    """
    # This number was chosen to be quieter than most of the audio samples in
    # the data set to avoid clipping.
    target_db=-24 
    rms = np.sqrt(np.mean(np.square(data)))
    data_db = librosa.amplitude_to_db(data)
    rms_db = librosa.amplitude_to_db([rms])[0]
    diff = target_db - rms_db
    data_db += diff
    return librosa.db_to_amplitude(data_db)

def recording_preprocess(samples, length_s=2.0, sr=16000):
    """Preprocess an audio signal recorded from the raspberry pi to be made
    into a spectrogram.

    This is the same processing that happens in data_preprocess, except no
    background noise is added.

    Parameters
    ----------
    samples: a numpy array of single-channel floating point audio data
    length_s: the number of seconds to shorten the preprocessed audio to.
        Defaults to 2.0s.
    sr: the sample rate, in Hz, of the audio clip. Defaults to 16000 Hz.

    Returns
    -------
    A numpy array of single channel floating point audio representing a
        shortened recording. The RMS loudness is normalized to -24dB.
    """
    shortened_clip = select_clip(samples, sr)
    return normalize_audio(shortened_clip)

def data_preprocess(samples, path_to_chunked_noise, length_s=2.0, sr=16000, loudness_scaling=0.2):
    """Preprocess audio from the dataset to be made into a spectrogram.

    Parameters
    ----------
    samples: a numpy array of single-channel floating point audio data.
    path_to_chunked_noise: the path to the directory containing short clips of
        background noise to add to the dataset.
    length_s: the number of seconds to shorten the preprocessed audio to.
        Defaults to 2.0s.
    sr: the sample rate, in Hz, of the audio clip. Defaults to 16000 Hz.
    loudness_scaling: the scale factor to apply to the background audio before
        adding it to samples.

    Returns
    -------
    A numpy array of single channel floating point audio representing a
        shortened clip from the dataset with background noise added. The
        loudness is normalized to -24dB.
    """
    shortened_clip = select_clip(samples, sr=sr)
    # Noise is added before normalization.
    noise_added = add_background_noise(shortened_clip, path_to_chunked_noise, loudness_scaling=loudness_scaling)
    return normalize_audio(noise_added)
