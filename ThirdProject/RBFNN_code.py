from time import time
from keras.engine.base_layer_v1 import Layer
from keras.initializers.initializers_v1 import RandomUniform
from keras.initializers.initializers_v2 import Initializer, Constant
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tensorflow import reduce_sum
from tensorflow.python.keras.backend import expand_dims, transpose, exp
import keras
import numpy as np
from keras.datasets import mnist


# Configuration options
feature_vector_length = 784
num_classes = 10

# Load the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape to 28 x 28 pixels = 784 features
X_train = X_train.reshape (X_train.shape [0], feature_vector_length)
X_test = X_test.reshape (X_test.shape [0], feature_vector_length)

# Standardize dataset
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.
X_test /= 255.

# Convert target classes to categorical ones
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# Set the input shape
input_shape = (feature_vector_length,)
print (f"Feature shape: {input_shape}")


class InitCentersRandom(Initializer):
    # """ Initializer for initialization of centers of RBF network
    # as random samples from the given data set.
    # Arguments
    # X: matrix, dataset to choose the centers from (random rows
    # are taken as centers)
    
    def __init__(self, X):
        self.X = X
        super().__init__()
    
    def __call__(self, shape, dtype=None):
        assert shape [1:] == self.X.shape [1:] # check dimension
        
        # np.random.randint returns ints from [low, high) !
        idx = np.random.randint(self.X.shape [0], size=shape [0])
        return self.x[idx, :]
    

class RBFLayer (Layer):
    def _init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        
        self.output_dim = output_dim

        # betas is either initializer object or float
        if isinstance (betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)
        
        self.initializer = initializer if initializer else RandomUniform(0.0, 1.0)
        
        super().__init__(**kwargs)


    def build(self, input_shape):

        self.centers = self.add_weight(
            name="centers",
            shape=(self.output_dim, input_shape [1]),
            initializer=self.initializer,
            trainable=True,
        )

        self.betas = self.add_weight(
            name="betas",
            shape=(self.output_dim,),
            initializer=self.betas_initializer,
            # initializer='ones',
            trainable=True,
        )
        
        super().build(input_shape)


    def call(self, x):
        
        C = expand_dims (self.centers, -1) # inserts a dimension of 1
        H = transpose(C - transpose(x)) # matrix of differences
        return exp(-self.betas * reduce_sum (H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config={"output_dim": self.output_dim}
        base_config = super().get_config()
        return dict(list (base_config.items()) + list(config.items()))


# create RBF network as keras sequential model
RBFlayer = RBFLayer(10, initializer=InitCentersRandom(X_train), betas=0.5, input_shape=input_shape) 
# RBFlayer RBFLayer (100, initializer-InitCenters KMeans (X_train), betas-0.1, input_shape=input_shape)
inputs = keras.layers.Input(input_shape) # input layer
x = RBFlayer (inputs) # hidden layer
outputs = Dense (10, activation="softmax") (x) # output layer

model = keras.models.Model(inputs, outputs)
print(input_shape)
print(model.summary())

# compile the model
model.compile(loss="mean_squared_error", optimizer=RMSprop(), metrics=["accuracy"])

# fit and predict
start = time()
h = model.fit(X_train, Y_train, batch_size=50, epochs=35, verbose=1, validation_split=0.2) 
train_time = time() - start

# Test the model after training
test_results = model.evaluate(X_test, Y_test, verbose=1)
print (f"Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}")

# Test the model after training
test_results = model.evaluate(X_test, Y_test, verbose=1)
print (f"Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}")

# accuracy values at the end of each epoch (if you have used `acc metric) print("accuracies: ")
print(h.history["accuracy"])

# training time
print("training time: ") 
print(train_time)

# list of epochs number
print(h.epoch)