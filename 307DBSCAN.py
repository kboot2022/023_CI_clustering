

#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import numpy as np

# The datafile in google drive root/Colab Notebooks/data/t4.8k_sample.txt
# for the full data, we can read it like below -

df = pd.read_csv("data/t4.8k.txt", sep=" ", names=['x','y'])

#df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/git_repos/2023_CI_clustering/data/t4.8k_sample.csv')

df.head()

df.shape

from sklearn.cluster import DBSCAN
# This is a model Dr. Song tuned with the sampled model.
db = DBSCAN(eps=15, min_samples=15, metric='euclidean')
labels = db.fit_predict(df)

df["cluster_lables"] = labels
df.head(2)

df.cluster_lables.value_counts()

# On HPC, this line should be changed, too!
df.to_csv("data/t4.8k_output.csv")



