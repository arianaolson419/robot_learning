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

model = keras.models.Sequential()
model.add(keras.layers.Dropout(0.2, input_shape=train_data[0].shape))
model.add(keras.layers.Conv2D(64, (5, 5), strides=(15, 10), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D((5, 5)))
model.add(keras.layers.Dropout(0.1))
#model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

model.fit(train_data, training_labels, validation_data=(valid_data, validation_labels), epochs=10)
#model.save('cnn.h5')
