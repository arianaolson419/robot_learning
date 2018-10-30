import tensorflow.keras as keras
from data_preprocess import partition_data, sample_info

train, valid = partition_data('../../Spectrograms', 0.9)
# Training data
training_labels = [sample_info(f) for f in train]
# Validation data
validation_labels = [sample_info(f) for f in validation]

inputs = keras.layers.Input(shape=(1025, 114, 1))
conv1 = keras.layers.Conv2D(64, (8, 8), strides=(10, 3), padding='same', activation='relu')(inputs)
conv2 = keras.layers.Conv2D(128, (8, 8), strides=(5, 2), padding='same', activation='relu')(conv1)
conv3 = keras.layers.Conv2D(256, (5, 5), strides=(3, 3), padding='same', activation='relu')(conv2)
conv4 = keras.layers.Conv2D(512, (7, 7), strides=(7, 7), padding='same', activation='relu')(conv3)
conv5 = keras.layers.Conv2D(2, (1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
flatten = keras.layers.Flatten()(conv5)
prediction = keras.layers.Dense(2, activation='softmax')(flatten)

model = keras.models.Model(inputs=inputs, outputs=prediction)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

model.fit(train, training_labels, validation_data=(valid, validation_labels), epochs=10)
