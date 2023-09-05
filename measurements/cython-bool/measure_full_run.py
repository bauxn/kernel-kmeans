from sklearn.datasets import make_blobs
from KKMeans.kernels import build_kernel_matrix
from KKMeans.utils import fill_empty_clusters
import numpy as np
from time import time
from KKMeans import KKMeans


class kkm_bool():
    '''dummy class to implement indexed simple lloyd's kernel kmeans'''
    def __init__(self, labels, n_clusters, rng, max_iter, tol, kernel, kernel_matrix):
        self.labels = labels
        self.n_clusters = n_clusters
        self.rng = np.random.default_rng(rng)
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.kernel_matrix = kernel_matrix
        
    def lloyd_bool(self):
        inertia = 0
        for _ in range(self.max_iter):
            distances = np.ascontiguousarray(np.tile(np.diag(self.kernel_matrix), (self.n_clusters, 1)).T)
            self._lloyd_iter(self.kernel_matrix, distances)
            labels_new = np.argmin(distances, axis=1)
            inertia_new = np.amin(distances, axis=1).sum()
            if all(labels_new == self.labels) or abs(inertia - inertia_new) < self.tol:
                break                
            self.labels = labels_new
            inertia = inertia_new

    def _lloyd_iter(self, kernel_matrix, distances):
        for cluster in range(self.n_clusters):
            mask = (self.labels == cluster)
            n_cluster_elements = sum(mask) 
            if n_cluster_elements == 0:
                self.labels[self.rng.integers(len(self.labels))] = cluster
                n_cluster_elements = 1      
            # (SUM K(a,b) for a,b in Cluster) / |cluster|**2 
            inner_term = kernel_matrix[mask][:, mask].sum() / (n_cluster_elements ** 2)
            # array that contains for each datapoint x: 2 * (SUM K(x,b) for b in Cluster) / |Cluster|
            element_term = 2 * kernel_matrix[:, mask].sum(axis = 1) / n_cluster_elements
            distances[:, cluster] += inner_term
            distances[:, cluster] -= element_term


runs = 10
max_iter = 300
tol = 0
rng = 0
kernel = "linear"
n_samples = 5000
n_features = 10
n_clusters = 100

kkm_store = list()
kkmb_store = list()

data, labels = make_blobs(n_samples, n_features, centers = n_clusters, random_state=rng)
km = build_kernel_matrix(data, kernel=kernel)

kkm = KKMeans(n_clusters, n_init=1, max_iter=max_iter, tol=tol, rng=rng, algorithm="lloyd", kernel=kernel)
labels = kkm.kmeanspp(data, km)
labels, _ = fill_empty_clusters(labels, n_clusters)

kkmb = kkm_bool(labels, n_clusters, rng, max_iter, tol, kernel, km)

for i in range(runs):
    start = time()
    kkmb.lloyd_bool()
    end = time()
    kkmb_store.append(end - start)
    print("midway through")
    start = time()
    kkm.lloyd(km, labels)
    end = time()
    kkm_store.append(end - start)
    

print("avg_cython:", sum(kkm_store) / len(kkm_store))
print("avg_bool:", sum(kkmb_store) / len(kkmb_store))

with open("full_run.txt", "a") as file:
    file.write("runs: " + str(runs) + "\n")
    file.write("max_iter: " + str(max_iter) + "\n")
    file.write("tol: " + str(tol) + "\n")
    file.write("rng: " + str(rng) + "\n")
    file.write("n_samples: " + str(n_samples) + "\n")
    file.write("n_features: " + str(n_features) + "\n")
    file.write("n_clusters: " + str(n_clusters) + "\n")
    file.write("\n")
    file.write("avg_cython:" + str(sum(kkm_store) / len(kkm_store)) + "\n")
    file.write("avg_bool:"+ str(sum(kkmb_store) / len(kkmb_store)) + "\n" + "\n")