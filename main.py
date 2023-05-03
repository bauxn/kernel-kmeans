from kernel_kmeans import KKMeans
from vis import visualize
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans

x, labels, centers = make_blobs(10000, centers = 10, return_centers = True, random_state = 0, n_features = 2)
kkm = KKMeans(n_clusters = 10, verbose = True, init = "kmeans++", kernel = "linear", random_state = 0, tol = 1e-4)
kkm.fit(x)
visualize(x, kkm.labels)