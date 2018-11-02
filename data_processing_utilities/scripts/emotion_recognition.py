from __future__ import print, division

import rospy
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

def class EmotionRecgonition(object):
    def __init__(self):
        rospy.init_node('emotion_recognition')
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.model = load_model('emotion_cnn.h5')

    def interpret_mic_input(self):
        """If the spectrogram is viable audio, detect emotion and make the robot move
        """
        pass
