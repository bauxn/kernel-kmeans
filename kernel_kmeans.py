import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

class KKMeans():
    def __init__(self, n_clusters = 8, init = "random", n_init = 1, 
                 max_iter = 300, tol = None, verbose = 0,
                 random_state = None, algorithm = "lloyd", kernel = "linear", **kwds):      
        self.n_clusters = n_clusters 
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.algorithm = algorithm
        self.kernel = kernel
        self.kwds = kwds
        self.labels = None
        return
    
    def fit(self, data):
        self._check_data(data)
        
        if self.algorithm == "lloyd":
            self._lloyd(data)
        
    def _check_data(self, data):
        return
    
        
    def _lloyd(self, data):
        self._init(data)
        kernel_matrix = pairwise_kernels(X = data, metric = self.kernel, **self.kwds)
        distances = np.zeros((len(data), self.n_clusters))
        
        for _ in range(self.max_iter):
            distances.fill(0)
            
            for cluster in range(self.n_clusters):
                mask = (self.labels == cluster)
                n_cluster_elements = sum(mask) #python sum faster on lists than np.sum? (SO says so. needs checking)
                if n_cluster_elements == 0:
                    if self.verbose:
                        print("Empty Cluster encountered. Assigned random element to cluster.")
                    self.labels[self.random_state.randint(len(self.labels))] = cluster
                    n_cluster_elements = 1      
                # (SUM K(a,b) for a,b in Cluster) / |cluster|**2 
                inner_term = kernel_matrix[mask][:,mask].sum() / (n_cluster_elements ** 2)
                # array that contains for each datapoint x: 2 * (SUM K(x,b) for b in Cluster) / |Cluster|
                element_term = 2 * kernel_matrix[:,mask].sum(axis = 1) / n_cluster_elements
                distances[:, cluster] += inner_term
                distances[:, cluster] -= element_term
            new_labels = np.argmin(distances, axis = 1)
            if self.tol:
                #todo
                break
            elif all(new_labels == self.labels):
                if self.verbose:
                    print("Converged at iteration: ", _ + 1)
                break
            self.labels = new_labels
        return

    def _init(self, data):

        if isinstance(self.init, (list, tuple, np.ndarray)):
            self._check_centroids() # TODO ? zumindest len(self.init) == self.n_clusters
            self._assign_to_centroids(self.init, data)
            return
        
        elif self.init == "random":
            centroids = data[self.random_state.randint(0, len(data), self.n_clusters)]
            print(self.n_clusters)
            print(centroids)
            self._assign_to_centroids(centroids, data)
            return
        
        elif self.init == "truerandom":
            self.labels = self.random_state.randint(0, self.n_clusters, len(data))
            return
        
        raise Exception("Unknown initialisation method")
            
    def _check_centroids(self):
        if len(self.init) != self.n_clusters:
            raise Exception("Is you dumb?") # TODO
        return

    def _assign_to_centroids(self, centroids, data):
        data_centr_kernel = pairwise_kernels(X = data, Y = centroids, metric = self.kernel, **self.kwds)
        centr_distances = np.zeros((len(data), self.n_clusters))
        for cluster in range(self.n_clusters):
            centr_distances[:, cluster] = -2 * data_centr_kernel[:, cluster] + pairwise_kernels([centroids[cluster]]) 
        self.labels = np.argmin(centr_distances, axis = 1)
        return

    def fit(self, data):
        self._check_data(data)
        if self.algorithm == "lloyd":
            self._lloyd(data)
        return
