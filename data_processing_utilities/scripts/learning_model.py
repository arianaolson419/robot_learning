import tensorflow.keras as keras
import numpy as np
from data_preprocess import partition_data, get_label, plot_spectrogram


def train_nn():

	train, valid = partition_data('../../Spectrograms', 0.9)
	# Training data

	train_data = np.expand_dims(np.array([np.load(t) for t in train]), axis=3)
	np.random.shuffle(train_data)
	training_labels = np.array([get_label(f) for f in train])
	#np.random.shuffle(training_labels)

	# Validation data
	valid_data = np.expand_dims(np.array([np.load(v) for v in valid]), axis=3)
	np.random.shuffle(valid_data)
	validation_labels = np.array([get_label(f) for f in valid])
	#np.random.shuffle(validation_labels)



	# model = keras.models.Sequential()
	# model.add(keras.layers.Dropout(0.1, input_shape=train_data[0].shape))
	# model.add(keras.layers.MaxPooling2D((5, 4)))
	# model.add(keras.layers.Conv2D(64, (5, 5), strides=(15, 10), padding='same', activation='relu'))
	# model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dense(2, activation='softmax'))

	# no maxpooling2d = 0.556%
	# maxpooling2d (5,2) = 0.585%
	# maxpooling2d (4,2) = 0.591%
	# maxpooling2d (2,2) = 0.620%
	# maxpooling2d (3,1) = 0.641%
	# maxpooling2d (3,3) = 0.650%
	# maxpooling2d (3,2) = 0.653%
	# maxpooling2d (4,2) = 0.658%
	# maxpooling2d (4,4) = 0.665%
	# maxpooling2d (5,4) = 0.676%
	# maxpooling2d (4,3) = 0.685%

	# dropout 0.2 = 0.613%
	# dropout 0.1 = 0.685%
	# dropout 0.05 = 0.652%

	# conv2d 32 = 0.647%
	# conv2d 64 = 0.685%
	# conv2d (5,5) = 0.685%
	# conv2d (4,5) = 0.608%

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

	# model = keras.models.Sequential()
	# model.add(keras.layers.Dropout(0.2, input_shape=train_data[0].shape))
	# model.add(keras.layers.Conv2D(64, (5, 5), strides=(15, 10), padding='same', activation='relu'))
	# model.add(keras.layers.MaxPooling2D((5, 5)))
	# model.add(keras.layers.Dropout(0.1))
	# #model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu'))
	# model.add(keras.layers.Flatten())
	# model.add(keras.layers.Dropout(0.2))
	# model.add(keras.layers.Dense(2, activation='softmax'))

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

if __name__ == '__main__':
	out = 0
	for i in range(1):
		out += train_nn()
	print(out/float(1))