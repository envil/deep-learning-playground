import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf

# tf.enable_eager_execution()
import tensorflow_probability as tfp
import numpy as np


def session_options(enable_gpu_ram_resizing=True):
    """Convenience function which sets common `tf.Session` options."""
    config = tf.ConfigProto()
    config.log_device_placement = True
    if enable_gpu_ram_resizing:
        # `allow_growth=True` makes it possible to connect multiple colabs to your
        # GPU. Otherwise the colab malloc's all GPU ram.
        config.gpu_options.allow_growth = True
    return config


def reset_sess(config=None):
    """Convenience function to create the TF graph and session, or reset them."""
    if config is None:
        config = session_options()
    tf.reset_default_graph()
    global sess
    try:
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession(config=config)


reset_sess()


# Target distribution is proportional to: `exp(-x (1 + x))`.
def unnormalized_log_prob(x):
    return -x - x ** 2.


# Create state to hold updated `step_size`.
step_size = tf.get_variable(
    name='step_size',
    initializer=1.,
    use_resource=True,  # For TFE compatibility.
    trainable=False)

# Initialize the HMC transition kernel.
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_log_prob,
    num_leapfrog_steps=3,
    step_size=step_size,
    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy())

# Run the chain (with burn-in).
samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=int(10e3),
    num_burnin_steps=int(1e3),
    current_state=1.,
    kernel=hmc)

# Initialize all constructed variables.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    init_op.run()
    samples_, kernel_results_ = sess.run([samples, kernel_results])

print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
    samples_.mean(), samples_.std(), kernel_results_.is_accepted.mean()))
# mean:-0.5003  stddev:0.7711  acceptance:0.6240

# plt.figure(figsize=(12, 6))
# plt.scatter(np.arange(np.size(samples_)), samples, s=4, alpha=0.6)
# # plotting the mean
# plt.show()
