import numpy as np
from sklearn.decomposition import PCA

def computeEigenFaces(data, k):
    ### data instances should be on dimension 0
    pca = PCA(k, whiten=True)
    pca.fit(data.reshape(data.shape[0], -1))
    e_vector = pca.components_
    e_values = pca.singular_values_
    return e_vector.reshape(k, *data.shape[1:]), e_values


