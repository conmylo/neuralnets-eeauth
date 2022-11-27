# import the libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the dataset
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(60000, 784).astype('float32')

# normalize the dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

# choose model properties like hidden layers, activation functions and number of neurons
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# fit
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))
