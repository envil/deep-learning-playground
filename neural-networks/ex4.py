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
TEST_DATA_RATIO = 0.8


def test_divider(batch_size):
    return int(TEST_DATA_RATIO * batch_size)


def main(number_of_neurons):
    print(f'Running for NN with', number_of_neurons, 'hidden neurons')
    # Defining input size, hidden layer size, output size and batch size respectively
    n_in, n_h, n_out, batch_size = 2, number_of_neurons, 1, 5000

    x_raw = np.pi * np.random.rand(batch_size, n_in) - np.pi / 2
    # y_raw = np.pi * np.random.rand(batch_size, n_in) - np.pi/2

    # Create training data
    x = torch.FloatTensor(x_raw[:test_divider(batch_size)])
    f = torch.FloatTensor(np.reshape(np.multiply(np.sin(x_raw[:test_divider(batch_size), 0]),
                                                 np.cos(x_raw[:test_divider(batch_size), 1])),
                                     (test_divider(batch_size), 1)))

    print(np.shape(x_raw[:test_divider(batch_size)]))

    # Create test data
    x_test = torch.FloatTensor(x_raw[test_divider(batch_size):])
    f_test = torch.FloatTensor(np.reshape(np.multiply(np.sin(x_raw[test_divider(batch_size):, 0]),
                                                      np.cos(x_raw[test_divider(batch_size):, 1])),
                                          (batch_size - test_divider(batch_size), 1)))

    # Create the first model
    model = nn.Sequential(nn.Linear(n_in, n_h),
                          nn.Tanh(),
                          nn.Linear(n_h, n_out),
                          nn.Tanh())

    # Construct the loss function
    criterion = torch.nn.MSELoss()

    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Gradient Descent
    for epoch in range(1200):
        # Forward pass: Compute predicted y by passing x to the model
        f_pred = model(x)

        # Compute and print loss
        loss = criterion(f_pred, f)
#         if epoch % 100 == 0:
#             print('epoch: ', epoch, ' loss: ', loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

    f_pred_test = model(x_test)
    test_loss = criterion(f_pred_test, f_test)

    print('Train loss: ', loss.item())
    print('Test loss : ', test_loss.item())
    fig = pyplot.figure(figsize=(15, 8))
    ax = fig.gca(projection='3d')

    x_draw, y_draw = np.meshgrid(x_raw[:, 0],
                                 x_raw[:, 1])
    x_draw_test, y_draw_test = np.meshgrid(x_raw[test_divider(batch_size):, 0],
                                           x_raw[test_divider(batch_size):, 1])
    z_draw_pred = f_pred.detach().numpy()
    z_draw_test = f_pred_test.detach().numpy() #np.multiply(np.sin(x_draw), np.cos(y_draw))
    surf_pred = ax.scatter(x_raw[:test_divider(batch_size), 0], x_raw[:test_divider(batch_size), 1], z_draw_pred, c='b', marker='^')
    surf_pred_test = ax.scatter(x_raw[test_divider(batch_size):, 0], x_raw[test_divider(batch_size):, 1], z_draw_test, c='r', marker='o')
    ax.legend((surf_pred, surf_pred_test), ('Train Result', 'Test Result'))
    pyplot.title('Result with {} hidden neurons'.format(number_of_neurons))
#     surf_test = ax.plot_surface(x_draw, y_draw, z_draw_test, cmap=cm.seismic, alpha=0.3,
#                                  linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(20))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    pyplot.show()


for x in range(1, 20, 2):
    main(x)
