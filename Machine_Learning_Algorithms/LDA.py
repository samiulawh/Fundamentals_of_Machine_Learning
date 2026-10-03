# Sami Ullah
# Roll no: 22013122-022

import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([
    [2,1,4,3], [3,1,1,9],
    [3,1,1,9], [3,1,1,9],
    [4,1,4,4], [3,1,1,9],
    [5,1,6,8], [3,1,1,9]
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, labels)

print("--- Cluster Labels ---")
print(labels)
print("\n--- LDA Projection (1D) ---")
print(X_lda)