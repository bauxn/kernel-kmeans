from KKMeans import KKMeans
from sklearn.datasets import make_blobs
from time import time

iters = 100
n_samples = 8000
n_clusters = 20
n_features = 5

time_store = list()

for seed in range(iters):
    kkm = KKMeans(n_clusters, "random", 1, tol=0, rng=seed)
    data, _ = make_blobs(n_samples, n_features, centers=n_clusters, random_state=seed)

    start = time()
    kkm.fit(data)
    end = time()
    time_store.append(end - start)

with open("openMP.txt", "a") as file:
    file.write("n_samples: " + str(n_samples) + "\n")
    file.write("time: " + str(sum(time_store) / len(time_store)))
    file.write("\n"+ "\n")