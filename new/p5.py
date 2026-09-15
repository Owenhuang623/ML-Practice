import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

# X = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]
# ]

# X, y = spiral_data(100, 3)

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activiation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: 
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities 

class Loss:
    def calculate(self, output, y):
        samples_losses = self.forward(output, y)
        data_loss = np.mean(samples_losses)
        return data_loss

class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_target.shape) == 1:
            correct_confidences = y_pred_clip[range(samples), y_target]
        elif len(y_target.shape) >= 1:
            correct_confidences = np.sum(y_pred_clip*y_target, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

X, y = spiral_data(samples = 100, classes=3)

### Architecture
dense1 = Layer_Dense(2, 3)
activation1 = Activiation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_Categorical_Cross_Entropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)