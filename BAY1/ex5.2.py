import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline
import daft
import seaborn

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

import tensorflow_probability as tfp

tfd = tfp.distributions
np.random.seed(5218)


def gmm(batch_size, n_clusters, alpha, sigma0):
    """ This is the solution for the process in 1.3 (only for 1-D data) """
    # parameters for Dirichlet distribution
    alpha = np.full(shape=(n_clusters,), fill_value=alpha, dtype='float32')

    # step 1: generate the assignment probability
    dirichlet = tfd.Dirichlet(concentration=alpha)
    theta = dirichlet.sample()

    # step 2: generate the centroid for each cluster
    normal_1 = tfd.Normal(loc=[0], scale=sigma0)
    # sampling `n_clusters` time, hence, mean for each cluster
    mu_k = normal_1.sample(n_clusters)  # (n_clusters, n_dim)
    print(mu_k)

    # ====== Now for the assignment, need 1 indicator for each
    # examples, hence we need `batch_size` amount of indicator
    # (step: 3(a))====== #
    categorical = tfd.OneHotCategorical(probs=theta)
    z = categorical.sample(batch_size)  # (batch_size, n_clusters)
    z = tf.cast(z, tf.bool)

    # ====== sampling the data points (step: 3(b)) ====== #
    normal_2 = tfd.Normal(loc=mu_k, scale=1)
    # this make each draw sample will generate sample for
    # all 4 components
    normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=2)
    x_all_components = normal_2.sample(batch_size)  # (batch_size, n_clusters, n_dim)
    # ====== selecting the right component for each sample (step: 3(b)) ====== #
    # (batch_size, n_clusters, n_dim) * (batch_size, n_clusters)
    # = (batch_size, n_dim)
    x = tf.boolean_mask(x_all_components, z)

    # ====== Return: X, Z, mu_k, theta ====== #
    return (x.numpy(),
            np.argmax(z.numpy(), axis=-1),
            mu_k.numpy(),
            theta.numpy())


X, Z, mu_k, theta = gmm(2500, n_clusters=3, alpha=1, sigma0=40)

plt.figure(figsize=(12, 6))
colors = seaborn.color_palette(palette='Set2', n_colors=len(np.unique(Z)))
plt.subplot(1, 2, 1)
# plotting the scatter points
plt.scatter(np.arange(len(X)), X, c=[colors[int(z)] for z in Z],
            s=4, alpha=0.6)
# plotting the mean
for i, mu in enumerate(mu_k):
    plt.axhline(mu, 0, len(X),
                label=r"$\mu=%.2f;\theta=%.2f$" % (mu_k[i], theta[i]),
                color=colors[i], linestyle='--', linewidth=2)
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
_ = plt.hist(X, bins=80)
plt.show()
