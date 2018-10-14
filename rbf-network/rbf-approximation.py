import numpy as np
from matplotlib import pyplot
# %matplotlib inline

# Defines how "wide" RBFs are (higher = RBF extends further)
BETA = 1

# Proportion of test data of all the data
PROPORTION_TEST = 0.3

# Prepare the data 
# Note: For simplicity, we do not do grid-space over x/y,
#       even though task could be interpreted so
x = np.linspace(0, 2 * np.pi, 30)
y = np.linspace(0, 2 * np.pi, 30)
z = np.sin(x) * np.cos(y)
# Combine x and y into one matrix
xy = np.vstack([x, y]).T

# Split into training and testing data
# 1. Create vector of indexes over all data
# 2. Randomly shuffle it
# 3. Take first N indexes for training
indexes = np.arange(len(x))
np.random.shuffle(indexes)
test_indexes = indexes[:int(len(indexes) * PROPORTION_TEST)]
train_indexes = indexes[int(len(indexes) * PROPORTION_TEST):]

# Create final training and testing set
# Training set will be of size
train_in = xy[train_indexes]
train_out = z[train_indexes]
test_in = xy[test_indexes]
test_out = z[test_indexes]


def pytorch_rbf(train_in, train_out, test_in, num_hidden=8, epochs=1000, lr=0.1, spread=1.0):
    """
    Implementing radial-basis network in Pytorch.
    I.e. Let PyTorch automatic-differentiation do all the magic
         required to train the parameters
    (Note: This is mainly for those who are interested in learning PyTorch)
    """
    import torch

    in_dim = train_in.shape[1]

    # Trainable parameters: Centers and weights
    # `requires_grad` tells PyTorch to compute gradients w.r.t these
    # variables
    weights = torch.randn(num_hidden, dtype=torch.double, requires_grad=True)
    centers = torch.randn(num_hidden, in_dim, dtype=torch.double, requires_grad=True)

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    test_in = torch.from_numpy(test_in)

    # Training loop
    outputs = None
    for epoch in range(epochs):
        # Do the predictions with current variables
        # as in lecture notes.
        # Using Gaussian kernel here
        # 1) Compute RBFs (#num_hidden)
        # 2) Do weighted sum of results
        outputs = torch.zeros(len(train_in), dtype=torch.double)
        for i in range(len(train_in)):
            squared_norm = torch.sum((train_in[i] - centers) ** 2, dim=1)
            # Gaussian kernel
            rbfs = torch.exp(-(squared_norm / spread ** 2))
            prediction = torch.dot(rbfs, weights)
            outputs[i] = prediction

        # Calculate SSE loss
        loss = (train_out - outputs).pow(2).sum()

        # Calculate gradients w.r.t loss
        # (Recall the variables we created with `requires_grad=True`)
        loss.backward()

        # Now variable ".grad" of weights and centers
        # will have gradients from taking gradient of the loss.
        # This will make sure we do not compute more gradients
        with torch.no_grad():
            # Apply learning step
            weights -= lr * weights.grad
            centers -= lr * centers.grad

            # We need to manually reset gradients for next iteration
            weights.grad.zero_()
            centers.grad.zero_()

    training_outputs = outputs

    # Do predictions for test set too
    test_outputs = torch.zeros(len(test_in), dtype=torch.double)
    for i in range(len(test_in)):
        squared_norm = torch.sum((test_in[i] - centers) ** 2, dim=1)
        rbfs = torch.exp(-(squared_norm / spread ** 2))
        prediction = torch.dot(rbfs, weights)
        test_outputs[i] = prediction

    # Bit of extra jargon here to get numpy arrays from PyTorch
    return (outputs.detach().numpy(), test_outputs.detach().numpy())


def debug_rbf(train_in, train_out, test_in):
    """
    Return random results
    """
    return [np.random.random((len(train_in),)), np.random.random((len(test_in),))]


# Train RB-network and get predictions for test set
# train_predictions, test_predictions = debug_rbf(train_in, train_out, test_in)
train_predictions, test_predictions = pytorch_rbf(
    train_in, train_out, test_in,
    num_hidden=16, epochs=1000, lr=0.01, spread=1.0
)

# Calculate errors and plot things
train_mse = np.mean((train_predictions - train_out) ** 2)
test_mse = np.mean((test_predictions - test_out) ** 2)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print("Train MSE:  %f" % train_mse)
print("Train RMSE: %f" % train_rmse)
print("Test MSE:   %f" % test_mse)
print("Test RMSE:  %f" % test_rmse)

pyplot.figure(dpi=200)
pyplot.plot(x, z)
pyplot.scatter(x[train_indexes], train_predictions)
pyplot.scatter(x[test_indexes], test_predictions)

pyplot.legend(("Original", "Train predictions", "Test predictions"))
pyplot.show()
