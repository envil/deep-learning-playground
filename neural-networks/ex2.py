import numpy as np

# Number of training iterations
NUM_ITERATIONS = 50
# Threshold for 0/1 classification
THRESHOLD = 0.0
# Learning rate
LEARNING_RATE = 0.01

# Data (both training and testing)
input_data = np.linspace(0, 5, 1000)


def output(n):
    return train_value_function(n[0]) + 0.5 * train_value_function(n[1]) - 1.5 * train_value_function(n[2])


def train_value_function(t):
    return np.sin(10 * np.sin(t) * t)


def sigmoid(x):
    return np.power((1 + np.exp(-x)), -1)


def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


# Create perceptron weights (random weights)
weights = np.random.rand(4)

# Train perceptron
for i in range(2, len(input_data)):
    # Calculate predictions with current weights
    # print(np.concatenate((input_data[i - 2:i+1], [1])))
    current_input = np.concatenate((input_data[i - 2:i + 1], [1]))
    predictions = np.dot(current_input, weights)

    output_value = tanh(train_value_function(current_input))

    # # Calculate accuracy (not needed for training, but to track the learning progress)
    print(predictions, output_value, train_value_function(current_input))
    accuracy = np.mean(predictions == output_value)
    # # Print the accuracy
    #     print("Iteration %d: Acc %f \t %s\n" % (i, accuracy, str(predictions)))
    #
    # # Update weights according to update rule
    weights = weights + LEARNING_RATE * (output_value - predictions) * current_input
    # print(weights)

# Print weights for inspection
print(weights)
