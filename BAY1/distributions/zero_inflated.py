# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ======================================================================== #
#
# Copyright (c) 2017 - 2018 scVAE authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================== #

"""The ZeroInflated distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import dtype_util

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from tensorflow import where

class ZeroInflated(distribution.Distribution):
  """zero-inflated distribution.

  The `zero-inflated` object implements batched zero-inflated distributions.
  The zero-inflated model is defined by a zero-inflation rate
  and a python list of `Distribution` objects.

  Methods supported include `log_prob`, `prob`, `mean`, `sample`, and
  `entropy_lower_bound`.
  """

  def __init__(self,
               dist,
               pi,
               validate_args=False,
               allow_nan_stats=True,
               name="ZeroInflated"):
    """Initialize a zero-inflated distribution.

    A `ZeroInflated` is defined by a zero-inflation rate
    (`pi`, representing the probabilities of excess zeros)
    and a `Distribution` object
    having matching dtype, batch shape, event shape, and continuity
    properties (the dist).

    Args:
      pi: A zero-inflation rate.
        Representing the probabilities of excess zeroes.

      dist: A `Distribution` instance.
        The instance must have `batch_shape` matching the zero-inflation rate.

      validate_args: Python `bool`, default `False`. If `True`, raise a runtime
        error if batch or event ranks are inconsistent between pi and any of
        the distributions. This is only checked if the ranks cannot be
        determined statically at graph construction time.

      allow_nan_stats: Boolean, default `True`. If `False`, raise an
       exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.

      name: A name for this distribution (optional).

    Raises:
      TypeError: If pi is not a zero-inflation rate, or `dist` is not
        `Distibution` are not
        instances of `Distribution`, or do not have matching `dtype`.
      ValueError: If `dist` is an empty list or tuple, or its
        elements do not have a statically known event rank.
        If `pi.num_classes` cannot be inferred at graph creation time,
        or the constant value of `pi.num_classes` is not equal to
        `len(dist)`, or all `dist` and `pi` do not have
        matching static batch shapes, or all dist do not
        have matching static event shapes.
    """
    parameters = dict(locals())
    if not dist:
      raise ValueError("dist must be non-empty")

    if not isinstance(dist, distribution.Distribution):
      raise TypeError(
          "dist must be a Distribution instance"
          " but saw: %s" % dist)

    # Ensure that all batch and event ndims are consistent.
    with ops.name_scope(name, values=[pi]):
      dtype = dtype_util.common_dtype([pi], preferred_dtype=tf.float32)
      pi = tf.convert_to_tensor(pi, name="pi", dtype=dtype)

      dtype = dist.dtype
      static_event_shape = dist.event_shape
      static_batch_shape = pi.get_shape()

      if static_event_shape.ndims is None:
        raise ValueError(
            "Expected to know rank(event_shape) from dist, but "
            "the distribution does not provide a static number of ndims")

      with ops.control_dependencies([check_ops.assert_positive(pi)] if
                                    validate_args else []):
        pi_batch_shape = array_ops.shape(pi)
        pi_batch_rank = array_ops.size(pi_batch_shape)
        if validate_args:
          dist_batch_shape = dist.batch_shape_tensor()
          dist_batch_rank = array_ops.size(dist_batch_shape)
          check_message = ("dist batch shape must match pi batch shape")
          self._assertions = [check_ops.assert_equal(
              pi_batch_rank, dist_batch_rank, message=check_message)]
          self._assertions += [
              check_ops.assert_equal(
                  pi_batch_shape, dist_batch_shape, message=check_message)]
        else:
          self._assertions = []

        self._pi = pi
        self._dist = dist
        self._static_event_shape = static_event_shape
        self._static_batch_shape = static_batch_shape

    # We let the zero-inflated distribution access _graph_parents since its arguably
    # more like a base class.
    graph_parents = [self._pi]  # pylint: disable=protected-access
    graph_parents += self._dist._graph_parents  # pylint: disable=protected-access

    super(ZeroInflated, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=graph_parents,
        name=name)

  @property
  def pi(self):
    return self._pi

  @property
  def dist(self):
    return self._dist

  def _batch_shape_tensor(self):
    return array_ops.shape(self._pi)

  def _batch_shape(self):
    return self._static_batch_shape

  def _event_shape_tensor(self):
    return self._dist.event_shape_tensor()

  def _event_shape(self):
    return self._static_event_shape

  def _mean(self):
    with ops.control_dependencies(self._assertions):
      # These should all be the same shape by virtue of matching
      # batch_shape and event_shape.
      return (1 - self._pi) * self._dist.mean()

  def _variance(self):
    with ops.control_dependencies(self._assertions):
      # These should all be the same shape by virtue of matching
      # batch_shape and event_shape.
      return (1 - self._pi) * (self._dist.variance() + math_ops.square(self._dist.mean())) - math_ops.square(self._mean())

  def _log_prob(self, x):
    with ops.control_dependencies(self._assertions):
      x = ops.convert_to_tensor(x, name="x")
      y_0 = math_ops.log(self.pi + (1 - self.pi) * self._dist.prob(x))
      y_1 = math_ops.log(1 - self.pi) + self._dist.log_prob(x)
      return where(x > 0, y_1, y_0)

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _sample_n(self, n, seed=None):
    with ops.control_dependencies(self._assertions):
      n = ops.convert_to_tensor(n, name="n")
      static_n = tensor_util.constant_value(n)
      n = int(static_n) if static_n is not None else n
      pi_samples = self.pi
      # pi_samples = self.pi.sample(n, seed=seed)

      static_samples_shape = pi_samples.get_shape()
      if static_samples_shape.is_fully_defined():
        samples_shape = static_samples_shape.as_list()
        samples_size = static_samples_shape.num_elements()
      else:
        samples_shape = array_ops.shape(pi_samples)
        samples_size = array_ops.size(pi_samples)
      static_batch_shape = self.batch_shape
      if static_batch_shape.is_fully_defined():
        batch_shape = static_batch_shape.as_list()
        batch_size = static_batch_shape.num_elements()
      else:
        batch_shape = self.batch_shape_tensor()
        batch_size = math_ops.reduce_prod(batch_shape)
      static_event_shape = self.event_shape
      if static_event_shape.is_fully_defined():
        event_shape = np.array(static_event_shape.as_list(), dtype=np.int32)
      else:
        event_shape = self.event_shape_tensor()

      # Get indices into the raw pi sampling tensor. We will
      # need these to stitch sample values back out after sampling
      # within the component partitions.
      samples_raw_indices = array_ops.reshape(
          math_ops.range(0, samples_size), samples_shape)

      # Partition the raw indices so that we can use
      # dynamic_stitch later to reconstruct the samples from the
      # known partitions.
      partitioned_samples_indices = data_flow_ops.dynamic_partition(
          data=samples_raw_indices,
          partitions=pi_samples,
          num_partitions=self.num_dist)

      # Copy the batch indices n times, as we will need to know
      # these to pull out the appropriate rows within the
      # component partitions.
      batch_raw_indices = array_ops.reshape(
          array_ops.tile(math_ops.range(0, batch_size), [n]), samples_shape)

      # Explanation of the dynamic partitioning below:
      #   batch indices are i.e., [0, 1, 0, 1, 0, 1]
      # Suppose partitions are:
      #     [1 1 0 0 1 1]
      # After partitioning, batch indices are cut as:
      #     [batch_indices[x] for x in 2, 3]
      #     [batch_indices[x] for x in 0, 1, 4, 5]
      # i.e.
      #     [1 1] and [0 0 0 0]
      # Now we sample n=2 from part 0 and n=4 from part 1.
      # For part 0 we want samples from batch entries 1, 1 (samples 0, 1),
      # and for part 1 we want samples from batch entries 0, 0, 0, 0
      #   (samples 0, 1, 2, 3).
      partitioned_batch_indices = data_flow_ops.dynamic_partition(
          data=batch_raw_indices,
          partitions=pi_samples,
          num_partitions=self.num_dist)
      samples_class = [None for _ in range(self.num_dist)]

      for c in range(self.num_dist):
        n_class = array_ops.size(partitioned_samples_indices[c])
        seed = distribution_util.gen_new_seed(seed, "ZeroInflated")
        samples_class_c = self.dist[c].sample(n_class, seed=seed)

        # Pull out the correct batch entries from each index.
        # To do this, we may have to flatten the batch shape.

        # For sample s, batch element b of component c, we get the
        # partitioned batch indices from
        # partitioned_batch_indices[c]; and shift each element by
        # the sample index. The final lookup can be thought of as
        # a matrix gather along lopiions (s, b) in
        # samples_class_c where the n_class rows correspond to
        # samples within this component and the batch_size columns
        # correspond to batch elements within the component.
        #
        # Thus the lookup index is
        #   lookup[c, i] = batch_size * s[i] + b[c, i]
        # for i = 0 ... n_class[c] - 1.
        lookup_partitioned_batch_indices = (
            batch_size * math_ops.range(n_class) +
            partitioned_batch_indices[c])
        samples_class_c = array_ops.reshape(
            samples_class_c,
            array_ops.conpi([[n_class * batch_size], event_shape], 0))
        samples_class_c = array_ops.gather(
            samples_class_c, lookup_partitioned_batch_indices,
            name="samples_class_c_gather")
        samples_class[c] = samples_class_c

      # Stitch back together the samples across the dist.
      lhs_flat_ret = data_flow_ops.dynamic_stitch(
          indices=partitioned_samples_indices, data=samples_class)
      # Reshape back to proper sample, batch, and event shape.
      ret = array_ops.reshape(lhs_flat_ret,
                              array_ops.conpi([samples_shape,
                                               self.event_shape_tensor()], 0))
      ret.set_shape(
          tensor_shape.TensorShape(static_samples_shape).conpienate(
              self.event_shape))
      return ret

  def entropy_lower_bound(self, name="entropy_lower_bound"):
    r"""A lower bound on the entropy of this zero-inflated model.

    The bound below is not always very tight, and its usefulness depends
    on the zero-inflated probabilities and the dist in use.

    A lower bound is useful for ELBO when the `ZeroInflated` is the variational
    distribution:

    \\(
    \log p(x) >= ELBO = \int q(z) \log p(x, z) dz + H[q]
    \\)

    where \\( p \\) is the prior distribution, \\( q \\) is the variational,
    and \\( H[q] \\) is the entropy of \\( q \\). If there is a lower bound
    \\( G[q] \\) such that \\( H[q] \geq G[q] \\) then it can be used in
    place of \\( H[q] \\).

    For a zero-inflated of distributions \\( q(Z) = \sum_i c_i q_i(Z) \\) with
    \\( \sum_i c_i = 1 \\), by the concavity of \\( f(x) = -x \log x \\), a
    simple lower bound is:

    \\(
    \begin{align}
    H[q] & = - \int q(z) \log q(z) dz \\\
       & = - \int (\sum_i c_i q_i(z)) \log(\sum_i c_i q_i(z)) dz \\\
       & \geq - \sum_i c_i \int q_i(z) \log q_i(z) dz \\\
       & = \sum_i c_i H[q_i]
    \end{align}
    \\)

    This is the term we calculate below for \\( G[q] \\).

    Args:
      name: A name for this operation (optional).

    Returns:
      A lower bound on the entropy of zero-inflated distribution.
    """
    with self._name_scope(name, values=[self.pi.logits]):
      with ops.control_dependencies(self._assertions):
        distribution_entropies = [d.entropy() for d in self.dist]
        pi_probs = self._pi_probs(log_probs=False)
        partial_entropies = [
            c_p * m for (c_p, m) in zip(pi_probs, distribution_entropies)
        ]
        # These are all the same shape by virtue of matching batch_shape
        return math_ops.add_n(partial_entropies)

  def _pi_probs(self, log_probs):
    """Get a list of num_dist batchwise probabilities."""
    which_softmax = nn_ops.log_softmax if log_probs else nn_ops.softmax
    pi_probs = which_softmax(self.pi.logits)
    pi_probs = array_ops.unstack(pi_probs, num=self.num_dist, axis=-1)
    return pi_probs
