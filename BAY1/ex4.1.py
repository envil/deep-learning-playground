import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
import tensorflow_probability as tfp

tfd = tfp.distributions

N_MEANS = 4
MEAN_STEP = 10
N_VARIANCE = 4
VARIANCE_STEP = 20
N_SAMPLES = 1000

means = np.linspace(0, (N_MEANS - 1) * MEAN_STEP, N_MEANS)
variances = np.linspace(1, (N_VARIANCE - 1) * VARIANCE_STEP + 1, N_VARIANCE)

plt.figure()

index = 1
for mean in means:
    for variance in variances:
        dist = tfd.Normal(loc=mean, scale=np.sqrt(variance))
        samples = dist.sample(N_SAMPLES).numpy()

        ax = plt.subplot(N_MEANS, N_VARIANCE, index)
        ax.scatter(np.arange(len(samples)), samples, s=4)
        ax.grid(True)
        plt.title("Mean:%.2f  Var:%.2f" % (mean, variance))
        index += 1

plt.tight_layout()
