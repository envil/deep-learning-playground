import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import seaborn

import pandas as pd

ds = pd.read_csv('fish_data.txt',
                 sep=";", decimal=',', encoding="Latin-1")
# remove all column that contain NaN values
ids = ds.apply(lambda x: np.all(x.notna()), axis=0)
ds = ds.iloc[:, ids.tolist()]
# we only take sample with "MITTAUSAIKA" = "LOPETUS"
selected_row = ds.MITTAUSAIKA == "LOPETUS"
ds = ds[selected_row]
print(ds.describe())

_ = ds.hist(figsize=(8, 8), bins=25)
_ = seaborn.pairplot(ds)