
import rospy
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from geometry_msgs.msg import Twist
import librosa
import numpy as np
from data_preprocess import select_clip
import os


NEGATIVE = -1
POSITIVE = 1
NEUTRAL = 0

class EmotionRecognition(object):
    def __init__(self):
        rospy.init_node('emotion_recognition')
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.model = load_model('cnn.h5')
        self.mood = NEUTRAL
        # self.pi = pxssh.pxssh()
        # if not s.login('192.168.17.201', 'pi', 'raspberry'):
        # 	print "SSH shession failed on login"

    def interpret_mic_input(self):
        """If the spectrogram is viable audio, detect emotion and make the robot move
        """
        os.system("bash recorder.sh")
        data, rate = librosa.load("recordings/recording_.wav", sr=16000, res_type='scipy')
        processed = recording_preprocess(data)
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

    def main(self):
    	r = rospy.Rate(0.45)
    	while not rospy.is_shutdown():
    		self.interpret_mic_input()
    		r.sleep()



if __name__ == '__main__':
	emotion_recognition = EmotionRecognition()
	emotion_recognition.main()

