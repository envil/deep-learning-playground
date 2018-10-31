import torch
import torch.nn as nn
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from matplotlib import cm

SEED = 5218
np.random.seed(SEED)
torch.manual_seed(SEED)
TRAIN_DATA_RATIO = 0.8

ages = [15, 15, 15, 18, 28, 29, 37, 37, 44, 50, 50, 60, 61, 64, 65, 65, 72, 75, 75, 82, 85, 91, 91, 97, 98, 125, 142,
        142, 147, 147, 150, 159, 165, 183, 192, 195, 218, 218, 219, 224, 225, 227, 232, 232, 237, 246, 258, 276, 285,
        300, 301, 305, 312, 317, 338, 347, 354, 357, 375, 394, 513, 535, 554, 591, 648, 660, 705, 723, 756, 768, 860,
        ]

weights = [21.66, 22.75, 22.3, 31.25, 44.79, 40.55, 50.25, 46.88, 52.03, 63.47, 61.13, 81, 73.09, 79.09, 79.51, 65.31,
           71.9, 86.1, 94.6, 92.5, 105, 101.7, 102.9, 110, 104.3, 134.9, 130.68, 140.58, 155.3, 152.2, 144.5, 142.15,
           139.81, 153.22, 145.72, 161.1, 174.18, 173.03, 173.54, 178.86, 177.68, 173.73, 159.98, 161.29, 187.07,
           176.13, 183.4, 186.26, 189.66, 186.09, 186.7, 186.8, 195.1, 216.41, 203.23, 188.38, 189.7, 195.31, 202.63,
           224.82, 203.3, 209.7, 233.9, 234.7, 244.3, 231, 242.4, 230.77, 242.57, 232.12, 246.7,
           ]
data_size = len(ages)


def test_divider(data):
    return int(TRAIN_DATA_RATIO * len(data))


x_raw_train = np.reshape(ages[:test_divider(ages)], (test_divider(ages), 1))
y_raw_train = np.reshape(weights[:test_divider(weights)], (test_divider(weights), 1))

x_raw_test = np.reshape(ages[test_divider(ages):], (data_size - test_divider(ages), 1))
y_raw_test = np.reshape(weights[test_divider(weights):], (data_size - test_divider(weights), 1))


def main(number_of_neurons):
    print(f'Running for NN with', number_of_neurons, 'hidden neurons')
    # Defining input size, hidden layer size, output size and batch size respectively
    n_in, n_h, n_out, batch_size = 1, number_of_neurons, 1, 5000

    # Create training data
    x_train = torch.FloatTensor(x_raw_train)
    y_train = torch.FloatTensor(y_raw_train)

    # Create test data
    x_test = torch.FloatTensor(x_raw_test)
    y_test = torch.FloatTensor(y_raw_test)

    # Create the first model
    model = nn.Sequential(nn.Linear(n_in, n_h),
                          nn.ReLU(),
                          nn.Linear(n_h, n_out),
                          nn.ReLU())

    # Construct the loss function
    criterion = torch.nn.MSELoss()

    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Gradient Descent
    for epoch in range(1200):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_train)

        # Compute and print loss
        loss = criterion(y_pred, y_train)
        if epoch % 100 == 0:
            print('epoch: ', epoch, ' loss: ', loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

    y_pred_test = model(x_test)
    test_loss = criterion(y_pred_test, y_test)

    print('Train loss: ', loss.item())
    print('Test loss : ', test_loss.item())


# for x in range(1, 20, 2):
#     main(x)

main(10)
