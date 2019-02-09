import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def gmm(batch_size, n_clusters, n_dim):
    """ This is the solution for the process in 1.3 """
    # parameters for Dirichlet distribution
    alpha = np.ones(shape=(n_clusters,))

    #     sigma = (np.random.rand(n_clusters, n_dim) * VARIANCE).astype('float32')
    sigma = np.repeat(np.random.randint(1, n_clusters, (1, n_dim)),
                      repeats=n_clusters, axis=0).astype('float32')
    loc = (np.random.rand(n_clusters, n_dim)).astype('float32')
    #     print("Alpha:", alpha)
    #     print("Sigma0:", sigma)
    #     print("loc", loc)

    dirichlet = tfd.Dirichlet(concentration=alpha)
    theta = dirichlet.sample(batch_size)
    normal_1 = tfd.Normal(loc=loc, scale=sigma)
    normal_1 = tfd.Independent(normal_1, reinterpreted_batch_ndims=2)
    mu_k = normal_1.sample()

    categorical = tfd.OneHotCategorical(probs=theta)
    z = categorical.sample()
    z = np.argmax(z, axis=1)

    mu_x = np.array(list(map(lambda x: np.asarray(mu_k[x, :]), z)))

    normal_2 = tfd.Normal(loc=mu_x, scale=1)
    normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=1)
    x = normal_2.sample()

    return x.numpy()


K = [2, 4, 16, 256]
for k in K:
    X = gmm(1000, n_clusters=k, n_dim=2)
    plt.figure()
    plt.title('k = {}'.format(k))
    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
