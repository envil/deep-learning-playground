import numpy as np
from matplotlib import pyplot
import random as r
# % matplotlib inline

# Defines how "wide" RBFs are (higher = RBF extends further)
BETA = 1

# Proportion of test data of all the data
PROPORTION_TEST = 0.3

# Prepare the data
x = np.linspace(0, 2 * np.pi, 30)
y = np.sin(x) * np.cos(x)

# Split into training and testing data
# 1. Create vector of indexes over all data
# 2. Randomly shuffle it
# 3. Take first N indexes for training
indexes = np.arange(len(x))
np.random.shuffle(indexes)
test_indexes = indexes[:int(len(indexes) * PROPORTION_TEST)]
train_indexes = indexes[int(len(indexes) * PROPORTION_TEST):]

# Create final training and testing set
train_in = x[train_indexes]
train_out = y[train_indexes]
test_in = x[test_indexes]
test_out = y[test_indexes]


def rbf(inputs, weights, centers):
    """ 
    Use weights and centers to implement a RBF, and
    feed inputs through it.
    Returns predictions
    """
    # Hardcoded for 1D inputs
    inputs = inputs[:, None, None]
    squared_norm = np.sum((inputs - centers.T) ** 2, axis=-2)
    # Gaussian kernel
    rbfs = np.exp(-(squared_norm / BETA ** 2))
    predictions = np.dot(rbfs, weights)

    return predictions


def deserialize_parameters(serialized, num_hidden, in_dim):
    """
    Deserialize parameters into weight and centers, based on given parameters
    """
    weights = serialized[:num_hidden]
    centers = serialized[num_hidden:].reshape(num_hidden, in_dim)
    return weights, centers


def evaluate_fitness(inputs, outputs, parameters, num_hidden, in_dim):
    """
    Return fitness of given paramters
    Fitness = negative of SSE
    """
    weight, centers = deserialize_parameters(parameters, num_hidden, in_dim)
    predictions = rbf(inputs, weight, centers)
    return -np.sum((outputs - predictions) ** 2)


def ga_rbf(train_in, train_out, test_in, num_hidden=8, generations=100,
           population_size=1000, best_to_keep=20, mutation_rate=0.05, print_every_gen=10):
    """
    Genetic algorithm to train a single-layer RBF network
    """
    # Hardcoded for 1D inputs
    in_dim = 1

    # Create random initial candidates
    candidates = [np.random.randn(num_hidden + num_hidden * in_dim) for i in range(population_size)]
    best_candidate = None
    best_fitness = -np.inf

    # Loop over generations
    for gen in range(generations):
        # Evaluate fitnesses
        candidate_fitnesses = [
            (candidate,
             evaluate_fitness(train_in, train_out, candidate, num_hidden, in_dim)
             ) for candidate in candidates
        ]
        # Sort based on fitness (first = best)
        candidate_fitnesses = sorted(candidate_fitnesses, key=lambda x: x[1], reverse=True)

        # Get best X candidates
        best_candidates = [x[0] for x in candidate_fitnesses[:best_to_keep]]

        # Create new population by crossovering and mutating
        # best candidates
        candidates.clear()
        for i in range(population_size):
            # Pick two random parents
            parent1 = r.choice(best_candidates)
            parent2 = r.choice(best_candidates)
            # Pick random crossover point
            crossover_point = r.randint(0, len(parent1))

            # Create "offspring"
            new_candidate = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Add random mutations
            # First: Select "genes" that should be mutated
            # Second: Select how much they should be mutated
            mutation_switch = np.random.random((len(new_candidate),)) < mutation_rate
            mutation_amount = np.random.randn(len(new_candidate))
            total_mutation = mutation_switch * mutation_amount

            # Apply mutation
            new_candidate += total_mutation

            # Add new candidate to the population
            candidates.append(new_candidate)

        # Select best candidate so far based on best fitness
        if best_fitness < candidate_fitnesses[0][1]:
            best_candidate = candidate_fitnesses[0][0]
            best_fitness = candidate_fitnesses[0][1]

        if (gen % print_every_gen) == 0:
            print("Gen %d, fitness %f" % (gen, best_fitness))

    weights, centers = deserialize_parameters(best_candidate, num_hidden, in_dim)
    # Get train and test predictions of the best candidate
    train_predictions = rbf(train_in, weights, centers)
    test_predictions = rbf(test_in, weights, centers)

    return train_predictions, test_predictions


def es_rbf(train_in, train_out, test_in, num_hidden=8, generations=100,
           population_size=100, best_to_keep=5, mutation_rate=0.05, print_every_gen=10):
    """
    Use evolution strategies to train a single-layer RBF network.
    Very similar to genetic algorithms, not so complex though.

    best_to_keep: How many best results will be used to pick the next point
    mutation_rate: The standard deviation of 
    """
    # Hardcoded for 1D inputs
    in_dim = 1

    # Create random initial candidate
    center = np.random.randn(num_hidden + num_hidden * in_dim)
    best_fitness = -np.inf

    # Loop over generations
    for gen in range(generations):
        # Generate bunch of candidates around the center.
        # Candidates are distributed around the center according
        # to normal distribution
        candidates = center + np.random.randn(population_size, center.shape[0]) * mutation_rate

        # Evaluate fitnesses of the candidate solutions
        candidate_fitnesses = [
            (candidate,
             evaluate_fitness(train_in, train_out, candidate, num_hidden, in_dim)
             ) for candidate in candidates
        ]

        # Sort based on fitness (first = best)
        candidate_fitnesses = sorted(candidate_fitnesses, key=lambda x: x[1], reverse=True)

        # Get best X candidates
        best_candidates = [x[0] for x in candidate_fitnesses[:best_to_keep]]

        ## Select best candidate so far based on best fitness
        # if best_fitness < candidate_fitnesses[0][1]:
        best_fitness = candidate_fitnesses[0][1]
        # New center is the mean point of best candidates
        center = np.array(best_candidates).mean(axis=0)

        if (gen % print_every_gen) == 0:
            print("Gen %d, fitness %f" % (gen, best_fitness))

    weights, centers = deserialize_parameters(center, num_hidden, in_dim)
    # Get train and test predictions of the best candidate
    train_predictions = rbf(train_in, weights, centers)
    test_predictions = rbf(test_in, weights, centers)

    return train_predictions, test_predictions


def debug_rbf(train_in, train_out, test_in):
    """
    Return random results
    """
    return [np.random.random((len(train_in),)), np.random.random((len(test_in),))]


# Train RB-network and get predictions for test set
# train_predictions, test_predictions = ga_rbf(
#    train_in, train_out, test_in, num_hidden=32, mutation_rate=0.001,
# )
train_predictions, test_predictions = es_rbf(
    train_in, train_out, test_in, num_hidden=32, generations=250, population_size=1000, mutation_rate=0.005,
    best_to_keep=1
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

pyplot.figure(dpi=150)
pyplot.plot(x, y)
pyplot.scatter(x[train_indexes], train_predictions)
pyplot.scatter(x[test_indexes], test_predictions)

pyplot.legend(("Original", "Train predictions", "Test predictions"))
pyplot.show()
