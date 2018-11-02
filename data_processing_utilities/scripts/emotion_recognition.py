
import rospy
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from geometry_msgs.msg import Twist
import librosa
import numpy as np
from data_preprocess import select_clip


NEGATIVE = -1
POSITIVE = 1
NEUTRAL = 0

class EmotionRecognition(object):
    def __init__(self):
        rospy.init_node('emotion_recognition')
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.model = load_model('cnn.h5')
        self.mood = NEUTRAL

    def interpret_mic_input(self):
        """If the spectrogram is viable audio, detect emotion and make the robot move
        """
        data, rate = librosa.load("../../AudioData/DC/h02.wav", sr=16000, res_type='scipy')
        processed = select_clip(data)
        S = np.abs(librosa.stft(processed))
        S = np.expand_dims([S], axis=3)
        out = self.model.predict(S)
        if out[0][0] > out[0][1]:
        	self.mood = NEGATIVE
        elif out[0][0] < out[0][1]:
        	self.mood = POSITIVE
        else:
        	self.mood = NEUTRAL
        self.act()

    def publish_positive(self):
    	msg = Twist()
    	msg.linear.x = 0.1
    	self.publisher.publish(msg)

    def publish_neutral(self):
    	msg = Twist()
    	self.publisher.publish(msg)

    def publish_negative(self):
    	msg = Twist()
    	msg.angular.z = 0.1
    	self.publisher.publish(msg)

    def act(self):
    	print(self.mood)
    	if self.mood == NEUTRAL:
    		self.publish_neutral()
    	elif self.mood == POSITIVE:
    		self.publish_positive()
    	elif self.mood == NEGATIVE:
    		self.publish_negative()


if __name__ == '__main__':
	emotion_recognition = EmotionRecognition()
	emotion_recognition.interpret_mic_input()

