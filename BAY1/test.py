import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

# Create a single trivariate Dirichlet, with the 3rd class being three times
# more frequent than the first. I.e., batch_shape=[], event_shape=[3].
alpha = [1, 1]
dist = tfd.Dirichlet(alpha)

a = dist.sample()  # shape: [4, 5, 3]
# print(a.data)
with tf.Session() as sess:
    sess.run(a)
    a.eval()

# # x has one sample, one batch, three classes:
# x = [.2, .3, .5]   # shape: [3]
# dist.prob(x)       # shape: []
#
# # x has two samples from one batch:
# x = [[.1, .4, .5],
#      [.2, .3, .5]]
# dist.prob(x)         # shape: [2]
#
# # alpha will be broadcast to shape [5, 7, 3] to match x.
# x = [[...]]   # shape: [5, 7, 3]
# dist.prob(x)  # shape: [5, 7]