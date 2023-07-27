import numpy as np
from sklearn.datasets import make_blobs
from KKMeans import KKMeans
from time import time

runs = 10
max_iter = 300
tol = 0
rng = 0
kernel = "linear"
n_init = 1

ls_n_samples = [2000, 8000]
ls_n_features = [5, 50]
ls_n_clusters = [5, 50, 200]





for n_samples in ls_n_samples:
    for n_features in ls_n_features:
        for n_clusters in ls_n_clusters:
            kkml_store = list()
            kkme_store = list()
            data, _ = make_blobs(n_samples, n_features, centers=n_clusters) 
            for iter in range(runs):
                kkml = KKMeans(n_clusters, n_init=n_init, tol=tol, rng=rng, algorithm="lloyd")
                kkme = KKMeans(n_clusters, tol=tol, n_init=n_init, rng=rng, algorithm="elkan")
                start = time()
                kkme.fit(data)
                end = time()
                kkme_store.append(end - start)
                start = time()
                kkml.fit(data)
                end = time()
                kkml_store.append(end - start)
            with open("measure.txt", "a") as file:
                file.write("n_samples: " + str(n_samples) + "\n")
                file.write("n_features: " + str(n_features) + "\n")
                file.write("n_clusters: " + str(n_clusters) + "\n")
                file.write("\n")
                file.write("avg elkan: "+ str(sum(kkme_store) / len(kkme_store)) + "\n")
                file.write("avg lloyd: " + str(sum(kkml_store) / len(kkml_store))+ "\n" + "\n")