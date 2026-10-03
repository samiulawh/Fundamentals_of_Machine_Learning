# Sami Ullah
# Roll no: 22013122-022

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X = np.array([
    [2,1,4,3], [3,1,1,9],
    [3,1,1,9], [3,1,1,9],
    [4,1,4,4], [3,1,1,9],
    [5,1,6,8], [3,1,1,9]
])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
pca = PCA(n_components=2)
y = pca.fit_transform(X)
print(y)