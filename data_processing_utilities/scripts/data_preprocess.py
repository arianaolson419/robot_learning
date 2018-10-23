"""We are using the SAVEE dataset, which consists of utterances of actors speaking 15 sentences for 7 different emotions. We are only training for 'positive' and 'negative' emotions, so we need to relabel the data and convert it to spectrograms."""

from scipy.io import wavfile
from spectrogram_maker import SpectrogramMaker
import numpy as np
import matplotlib.pyplot as plt

import re

SAVEE_EMOTION_FULL_NAME = {'a': 'anger', 'd': 'disgust', 'f': 'fear', 'h': 'happiness', 'n': 'neutral', 'sa': 'sadness', 'su': 'surprise'}

# Negative emotions are given a 0 and positive emotions are given a 1.
POS_NEG_EMOTION_MAPPING = {'a': 0, 'd': 0, 'f': 0, 'h': 1, 'n': 1, 'sa': 0, 'su': 1}

def sample_info(filename):
    """
    Returns the emotion and sentence number of a given file from the SAVEE dataset.
    """
    filename = filename[:-4]
    emotion_label = re.split("[^a-z]+", filename)[0]
    sentence_label = filename[len(emotion_label):]
    return emotion_label, sentence_label

if __name__ == "__main__":
    example_file = '/home/ariana/catkin_ws/src/robot_learning/AudioData/DC/a01.wav'
    spectrogram_maker = SpectrogramMaker('/home/ariana/catkin_ws/src/robot_learning/AudioData', '/home/ariana/catkin_ws/src/robot_learning/Spectrograms')
    spectrogram = spectrogram_maker.make_spectrogram(example_file)
    plt.imshow(spectrogram[2].transpose())
    plt.show()
    print(spectrogram[2].shape)
    print(sample_info("ap01.wav"))
