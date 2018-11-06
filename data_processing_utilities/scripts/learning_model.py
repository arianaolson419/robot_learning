"""This script is for training a learning model using the spectrograms of clips
from the dataset.

Authors: Anna Buchele, Ariana Olson

Usage
-----
* Create spectrograms using spectrogram_maker.py
    python spectrogram_maker.py
Remember to change the source and destination directories for each speaker in
the dataset as per the instructions in spectrogram_maker.
* Train the network
    python learning_model.py
"""
import tensorflow.keras as keras
import numpy as np
from data_preprocess import partition_data, get_label, plot_spectrogram
from pyAudioAnalysis import audioTrainTest as aT


def train_nn(randomize_labels = False):

	train, valid = partition_data('../../Spectrograms', 0.9)
	# Training data

	train_data = np.expand_dims(np.array([np.load(t) for t in train]), axis=3)
	np.random.shuffle(train_data)
	training_labels = np.array([get_label(f) for f in train])

	# Validation data
	valid_data = np.expand_dims(np.array([np.load(v) for v in valid]), axis=3)
	np.random.shuffle(valid_data)
	validation_labels = np.array([get_label(f) for f in valid])

	if randomize_labels: 
		# To test if the nn is doing any better than purely random
		np.random.shuffle(training_labels)
		np.random.shuffle(validation_labels)

	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=train_data[0].shape))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(2, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.summary()

	model.fit(train_data, training_labels, validation_data=(valid_data, validation_labels), epochs=5)
	model.save('cnn.h5')

	valid_predicted = model.predict(valid_data)

	# for i in range(15):
	# 	print(valid_predicted[i])
	# 	print(validation_labels[i])
	# 	plot_spectrogram(valid_data[i])

	hist_pred = [(i[0] < i[1]) for i in valid_predicted]
	hist_valid = [(i[0] < i[1]) for i in validation_labels]
	num_correct = 0
	num_wrong = 0
	for i in range(len(valid_predicted)):
		if hist_pred[i] == hist_valid[i]:
			num_correct += 1
		else:
			num_wrong += 1
	print("Number of correct answers: %d \n Number of wrong answers: %d \n Percent accuracy: %f" % (num_correct, num_wrong, num_correct/float(len(valid_predicted))))
	return num_correct/float(len(valid_predicted))

def nonlearning():
	"""
	Using pyAudioAnalysis package to train parameters for classification of emotion. Sorted chunked data into positive and negative categories, 
	then followed the documentation here: https://github.com/tyiannak/pyAudioAnalysis/wiki/4.-Classification-and-Regression
	"""
	aT.featureAndTrain(['../../AudioData/chunked_data_sorted/pos', '../../AudioData/chunked_data_sorted/neg'], 
						1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, 
                    "svm", "emotion_classifier", True)

if __name__ == '__main__':
	nonlearning()
