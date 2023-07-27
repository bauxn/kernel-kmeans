from sklearn.datasets import make_blobs
from kernels import build_kernel_matrix as bkm
from sklearn.metrics.pairwise import pairwise_kernels as pk
from time import time
import numpy as np

runs = 10
n_samples = 1000
n_features = 50
kernel = "rbf"


cython_store = list()
sklearn_store = list()

for it in range(runs):
    data, labels = make_blobs(n_samples, n_features)
    start = time()
    b = bkm(data, kernel=kernel, gamma=1/n_features)
    end = time()
    cython_store.append(end - start)
    start = time()
    c = pk(data, metric=kernel)
    end = time()
    sklearn_store.append(end - start)
    assert np.allclose(b, c)
    

print("avg_cython:", sum(cython_store) / len(cython_store))
print("avg_bool:", sum(sklearn_store) / len(sklearn_store))

with open("outer_sum.txt", "a") as file:
    file.write("kernel: " + str(kernel) + "\n")
    file.write("n_samples: " + str(n_samples) + "\n")
    file.write("n_features: " + str(n_features) + "\n")
    file.write("\n")
    file.write("avg_cython:" + str(sum(cython_store) / len(cython_store)) + "\n")
    file.write("avg_sklearn:"+ str(sum(sklearn_store) / len(sklearn_store)) + "\n" + "\n")