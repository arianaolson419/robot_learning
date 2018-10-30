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

inputs = keras.layers.Input(shape=(1025, 63, 1))
conv1 = keras.layers.Conv2D(512, (5, 5), strides=(15, 5), padding='same', activation='relu')(inputs)
pool1 = keras.layers.MaxPooling2D((10, 10))(conv1)
flat = keras.layers.Flatten()(conv1)
prediction = keras.layers.Dense(2, activation='softmax')(flat)

model = keras.models.Model(inputs=inputs, outputs=prediction)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

model.fit(train_data, training_labels, validation_data=(valid_data, validation_labels), epochs=10)
#model.save('cnn.h5')
