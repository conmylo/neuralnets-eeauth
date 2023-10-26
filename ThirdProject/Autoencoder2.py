from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras import Input, Model
from keras.datasets import mnist
import numpy as np


model = Sequential()

# encoder network
model.add(Conv2D(30, 3, activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, padding='same'))
model.add(Conv2D(15, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, padding='same'))

# decoder network
model.add(Conv2D(15, 3, activation='relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(30, 3, activation='relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(1, 3, activation='sigmoid', padding='same'))

# output layer
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
model.fit(x_train, x_train, epochs=15, batch_size=128, validation_data=(x_test, x_test))