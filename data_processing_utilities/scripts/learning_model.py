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
from data_preprocess import partition_data, get_label

train, valid = partition_data('../../Spectrograms', 0.9)
# Training data
train_data = np.expand_dims(np.array([np.load(t) for t in train]), axis=3)
training_labels = np.array([get_label(f) for f in train])
# Validation data
valid_data = np.expand_dims(np.array([np.load(v) for v in valid]), axis=3)
validation_labels = np.array([get_label(f) for f in valid])

# Define the network.
model = keras.models.Sequential()
model.add(keras.layers.Dropout(0.2, input_shape=train_data[0].shape))   # Dropout layers are used to prevent overfitting.
model.add(keras.layers.Conv2D(32, (5, 5), strides=(15, 10), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D((5, 5)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Train and save the network.
model.fit(train_data, training_labels, validation_data=(valid_data, validation_labels), epochs=10)
# Uncomment below to save the model.
#model.save('cnn.h5')
