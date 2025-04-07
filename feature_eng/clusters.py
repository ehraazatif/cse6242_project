import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#load data
df_embed = pd.read_csv("../data_versions/embeddings.csv")
#turns str to py list
df_embed["Embedding"] = df_embed["Embedding"].apply(ast.literal_eval)
#turn to 2D np array
X = np.vstack

print("Shape Of Embedding Array:", X.shape)

#hierarchical clustering

#linkage matrix: Wards Method
Z = linkage(X, method = 'ward', metric = 'euclidean')

#dendogram
plt.figure(figsize = (10,5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples (Descriptions)")
plt.ylabel("Distance")
plt.show()

#cluster forming

DIST_THRESHOLD = #discuss with team
clusters = fcluster(Z, t = DIST_THRESHOLD, criterion = 'distance')

df_embed["Cluster"] = clusters

df_embed.sample(10)


