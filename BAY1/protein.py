from math import log1p

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from mpmath.math2 import EPS
from tensorflow.contrib.framework import is_tensor

tf.enable_eager_execution()

from tensorflow_probability.python.distributions import NegativeBinomial

def log_norm(X, axis=1, scale_factor=10000):
    """ Seurat log-normalize
    y = log(X / (sum(X, axis) + epsilon) * scale_factor)

    where log is natural logarithm
    """
    if is_tensor(X):
        return tf.log1p(
            X / (tf.reduce_sum(X, axis=axis, keepdims=True) + EPS) * scale_factor)
    elif isinstance(X, np.ndarray):
        X = X.astype('float64')
        return np.log1p(
            X / (np.sum(X, axis=axis, keepdims=True) + np.finfo(X.dtype).eps) * scale_factor)
    elif isinstance(X, pd.DataFrame):
        X = X.astype('float64')
        return X.apply(lambda x: log1p(x/(X.sum() + EPS) * scale_factor))
    else:
        raise ValueError("Only support numpy.ndarray or tensorflow.Tensor")


protein = pd.read_csv('protein.csv', sep=",")
# cd20 = protein['CD20']
# cd45 = protein['CD45']
# cd34 = protein['CD34']
# cd10 = protein['CD10']
# cd19 = protein['CD19']
# print(protein[protein.columns[1:]])
# protein_np = np.array([cd20, cd45, cd34, cd10, cd19])
print(protein.describe())
protein_norm = log_norm(protein.iloc[1:])
# print(protein_norm)
# protein_norm.hist(figsize=(16, 16), bins=25)
# seaborn.distplot(protein_norm)
# fig = cd20.hist(figsize=(16, 16), bins=50)
# fig.show()


