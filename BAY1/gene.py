import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow_probability.python.distributions import NegativeBinomial
import BAY1.distributions.zero_inflated as zi

# protein = genfromtxt('gene.csv', delimiter=',', skip_header=1, usecols=118)
protein = pd.read_csv('gene.csv', sep=",")

# remove all column that contain NaN values
ids = protein.where(lambda x: x.notna(), lambda x: int(0), axis=0)
# print(ids)
# ds = protein.iloc[:, ids.tolist()]

adap1 = protein['ADAP1']

_ = adap1.hist(figsize=(8, 8), bins=25)
# _ = seaborn.pairplot(protein)

fig, axes = plt.subplots(nrows=2, ncols=2)
# df1.hist(ax=axes[0,0])
# df2.plot(ax=axes[0,1])
adap1.hist(figsize=(16, 16), bins=50, ax=axes[0, 0])
fig.show()

nb = NegativeBinomial(total_count=adap1, probs=0.5)
zinb = zi.ZeroInflated(dist=nb, pi=0.9)

print(nb.mean())
print()

# matching values of imputed data
print(zinb.mean())
print(zinb.log_prob(10))