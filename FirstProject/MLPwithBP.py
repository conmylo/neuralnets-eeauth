# import the libraries
import numpy as np
from random import random


# MLP Class
class MLP(object):

    # Here we initialize the MLP, giving the numbers of inputs, the number of hidden
    # layers and the number of outputs
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create general representation of layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random weights of connections between layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # store the derivatives of each layer (we need these for the back propagation algorithm)
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    # function to forward propagate the MLP
    def forward_propagate(self, inputs):

        # the input layer activation is just the input itself
        activations = inputs

        # store the activations for back propagation algorithm
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # store activations for back propagation algorithm
            self.activations[i + 1] = activations

        return activations

    # back propagation algorithm function
    def back_propagate(self, error):

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # get activation of previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape into 2d
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations to 2d matrix
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            # store derivative after matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # back propagate the next error
            error = np.dot(delta, self.weights[i].T)

    # function to train the model
    def train(self, inputs, targets, epochs, learning_rate):

        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # update the weight by applying gradient descent on derivatives
                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)

            # print error per epoch
            print("Error: {} at epoch {}".format(sum_errors / len(items), i + 1))

        print("Training done")

    # descends the gradient to help the training process
    def gradient_descent(self, learningRate=1):

        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    # sigmoid function used for activation
    def _sigmoid(self, x):

        y = 1.0 / (1 + np.exp(-x))
        return y

    # sigmoid derivative function used for error
    def _sigmoid_derivative(self, x):

        return x * (1.0 - x)

    # mse function (mean squared error)
    def _mse(self, target, output):

        return np.average((target - output) ** 2)


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    items = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train the model
    mlp.train(items, targets, 100, 0.1)

    # testing set
    input = np.array([0.4, 0.3])
    target = np.array([0.7])

    # predict
    output = mlp.forward_propagate(input)

    # testing the model
    print()
    print("{} + {} is equals {}".format(input[0], input[1], output[0]))
