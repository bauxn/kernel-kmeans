import numpy as np
from utils import _calc_inner_sums
from kernels import build_kernel_matrix
from sklearn.datasets import make_blobs
from time import time



def inner_sum_bool(kernel_matrix, labels, n_clusters):
    inner_sums = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = (labels==c)
        inner_sums[c] = kernel_matrix[mask][:, mask].sum()
    return inner_sums

runs = 100
n_samples = 10000
n_features = 10
n_clusters = 100
rng = 0

data, labels = make_blobs(n_samples, n_features, centers = n_clusters, random_state=rng)
km = build_kernel_matrix(data)

bool_store = list()
cython_store = list()

for it in range(runs):
    start = time()
    b = inner_sum_bool(km, labels, n_clusters)
    end = time()
    bool_store.append(end - start)
    start = time()
    c = _calc_inner_sums(km, labels, n_clusters)
    end = time()
    cython_store.append(end - start)
    assert np.allclose(b, c)
    

print("avg_cython:", sum(cython_store) / len(cython_store))
print("avg_bool:", sum(bool_store) / len(bool_store))

with open("inner_sum.txt", "a") as file:
    file.write("runs: " + str(runs) + "\n")
    file.write("n_samples: " + str(n_samples) + "\n")
    file.write("n_features: " + str(n_features) + "\n")
    file.write("n_clusters: " + str(n_clusters) + "\n")
    file.write("\n")
    file.write("avg_cython:" + str(sum(cython_store) / len(cython_store)) + "\n")
    file.write("avg_bool:"+ str(sum(bool_store) / len(bool_store)) + "\n" + "\n")