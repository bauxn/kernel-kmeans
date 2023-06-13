import numpy as np
from time import time
from lloyd_utils import lloyd_update, calc_sq_distances, calc_sizes
from kernel_utils import kernel_matrix
from quality_utils import calc_silhouettes
from elkan_utils import update_elkan
start_elkan = lloyd_update  #TODO 

class KKMeans():
    def __init__(self, n_clusters=8, init="random", n_init=3,
                 max_iter=300, tol=0, q_metric="inertia", verbose=False,
                 seed=None, algorithm="lloyd", kernel="linear", **kwargs):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.q_metric = q_metric
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.algorithm = algorithm
        self.kernel = kernel
        self.kwargs = kwargs
        self.labels = None
        self.inner_sums = np.zeros(n_clusters, dtype=np.int_)
        self.cluster_sizes = None
        self.trained_data = None
        self.quality = None
    
    def kernel_wrapper(self, X, Y=None):
        return kernel_matrix(X, Y, kernel=self.kernel, **self.kwargs)

    def fit(self, X):
        
        X = self._sanitize_data(X)
        self._check_params(X)
        kernel_matrix = self.kernel_wrapper(X)
        
        labels_store = np.zeros((self.n_init, X.shape[0]), dtype=np.int_)
        quality_store = np.zeros(self.n_init, dtype=np.double)
        inner_sums_store = np.zeros((self.n_init, self.n_clusters), dtype=np.double)
        sizes_store = np.zeros((self.n_init, self.n_clusters), dtype=np.int_)
        
        for init in range(self.n_init):
            start_labels = self._init_labels(X, kernel_matrix)
            if self.algorithm == "lloyd":
                labels, quality, inner_sums, sizes = self._lloyd(kernel_matrix, start_labels)
            elif self.algorithm == "elkan":
                labels, quality, inner_sums, sizes = self._elkan(kernel_matrix, start_labels)
            else:
                raise ValueError(self.algorithm + " algorithm not implemented")
            labels_store[init] = labels
            quality_store[init] = quality
            inner_sums_store[init] = inner_sums
            sizes_store[init] = sizes

        best_init = self._get_best_init(quality_store)
        self.labels = labels_store[best_init]
        self.inner_sums = inner_sums_store[best_init]
        self.cluster_sizes = sizes_store[best_init]
        self.quality = quality_store[best_init]
        self.trained_data = X
        
        if self.verbose:
            print("Best " + self.q_metric +":", self.quality,
                "Found at init:", best_init + 1)

    def _check_params(self, data):
        if self.algorithm == "elkan" and self.q_metric == "silhouette":
            print("WARNING: using silhouette as metric with elkan will most likely be inaccurate\
                  as elkan  does not calculate exact distances to every center")
        if self.n_init <= 0:
            raise ValueError("n_inits needs to be at least 1")

        if data.shape[0] <  self.n_clusters:
            raise ValueError("sample:cluster ratio needs to be at least one")
        


        
    def _get_best_init(self, quality_store):
        if self.q_metric == "inertia":
            return np.argmin(quality_store)
        elif self.q_metric == "silhouette":
            return np.argmax(quality_store)
        else:
            raise Exception("Quality metric not implemented. This should never occur.")


    def _sanitize_data(self, X):
        X = np.asarray(X, dtype=np.double)
        if len(X.shape) != 2:
            raise ValueError("X needs to be 2-d Array")
        if 0 in X.shape:
            raise ValueError("X is empty")
        return X
    
    def _init_labels(self, X, kernel_matrix):
        '''Assign labels to each datapoint by given method'''
        if isinstance(self.init, (list, tuple, np.ndarray)):
            self.init = self._sanitize_centers() 
            return self._assign_to_centers(X, self.init)
        
        elif self.init == "random":
            centers = self.rng.choice(X, self.n_clusters)
            return self._assign_to_centers(X, centers)
        elif self.init == "truerandom":
            return self.rng.integers(0, self.n_clusters, len(X), dtype=np.int_)
        
        elif self.init == "kmeans++":
            return self._kmeanspp(X, kernel_matrix)
        
        raise NotImplementedError("Unknown initialisation method")
    
    def _sanitize_centers(self, centers):
        centers = np.asarray(centers, dtype=np.double)
        if len(centers.shape) != 2:
            raise ValueError("Given centers need to be 2-d array")
        if 0 in centers.shape:
            raise ValueError("Given centers are empty")
        if len(centers) != self.n_clusters:
            raise ValueError("Amount of given centers must be equal to n_clusters")
        return centers
        
    def _assign_to_centers(self, X, centers):
        X_center_kernel = self.kernel_wrapper(X, centers)
        dists_to_centers = np.zeros((len(X), self.n_clusters))
        for cluster in range(self.n_clusters):
            dists_to_centers[:, cluster] = (-2 * X_center_kernel[:, cluster]
                             + self.kernel_wrapper(centers[cluster]))
        return np.array(np.argmin(dists_to_centers, axis=1), dtype=np.int_)

    def _kmeanspp(self, X, kernel_matrix):
        dists_to_centers = self._build_starting_distance(kernel_matrix)
        data_size = X.shape[0]
        for cluster in range(self.n_clusters):
            if cluster == 0:
                index = self.rng.integers(low=0, high=data_size)
            else:
                max_dist_each = np.amin(dists_to_centers[:, :cluster + 1], axis = 1)
                probs = max_dist_each/max_dist_each.sum()
                index = self.rng.choice(len(X), size=1, p=probs)
            center = X[index]
            inner_sum = self.kernel_wrapper(center)
            outer_sum = self.kernel_wrapper(X, center)
            # reshape necessary as kernel_wrapper has 2dim array output
            dists_to_centers[:, cluster] += (-2 * outer_sum + inner_sum).reshape(data_size,)
            dists_to_centers[:, cluster] = np.sqrt(dists_to_centers[:, cluster])
             
            
        return np.array(np.argmin(dists_to_centers, axis=1), dtype=np.int_)

    def _lloyd(self, kernel_matrix, labels):
        quality = 0
        for it in range(self.max_iter):
            distances = self._build_starting_distance(kernel_matrix)
            distances, inner_sums, cluster_sizes =\
                    lloyd_update(distances, kernel_matrix, labels, self.n_clusters)
            labels_old = labels
            labels = np.argmin(distances, axis=1)
            
            quality, converged = self.measure_iter(distances, labels, labels_old, quality)
            self._out_verbose(it, quality, converged=converged)
            if converged:
                break
        
        return labels, quality, inner_sums, cluster_sizes
    


    def _out_verbose(self, iter, quality, converged):
        if not self.verbose:
            return
        if converged:
            print("Converged at iteration:", iter + 1, self.q_metric + ":", quality)
        elif self.tol == 0:
            print("Iteration:", iter + 1, self.q_metric + ":", "Not calculated (Tol==0)")
        else:
            print("Iteration:", iter + 1, self.q_metric + ":", quality)
    
    def predict(self, X):
        kernel_matrix = self.kernel_wrapper(X, self.trained_data)
        sq_distances = calc_sq_distances(self.inner_sums, 
                                  self.cluster_sizes,
                                  kernel_matrix,
                                  self.labels, 
                                  self.n_clusters)
        return np.argmin(np.sqrt(sq_distances), axis=1)
    
    def measure_iter(self, sq_distances, labels, labels_old, quality):
        converged = False
        if self.tol != 0:
            quality_old = quality
            quality = self._calc_q_metric(sq_distances, labels)
            if abs(quality - quality_old) <= self.tol:
                converged = True

        if all(labels_old == labels):
            if self.tol == 0:
                quality = self._calc_q_metric(sq_distances, labels)
            converged = True
        
        return quality, converged

    def _calc_q_metric(self, sq_distances, labels):
        if self.q_metric == "inertia":
            return sq_distances[range(sq_distances.shape[0]), labels].sum()
        elif self.q_metric == "silhouette":
            return self.calc_silhouette(sq_distances, labels)
        else:
            raise NotImplementedError(str(self.q_metric) + " quality metric not implemented")
            
    def calc_silhouette(self, sq_distances, labels):
        dists = np.sqrt(sq_distances)
        silhouettes = calc_silhouettes(dists, labels)
        return sum(silhouettes) / len(silhouettes)
    
    def _build_starting_distance(self, kernel_matrix):
        return np.ascontiguousarray(np.tile(np.diag(kernel_matrix), (self.n_clusters, 1)).T, dtype=np.double)

    def _elkan(self, kernel_matrix, labels):
        labels = np.array(labels, dtype=np.int_) # TODO delete and change dtype in init_labels
        start_dists = self._build_starting_distance(kernel_matrix)
        center_dists = np.zeros((kernel_matrix.shape[0], self.n_clusters))
        quality = 0
        for it in range(self.max_iter):
            if it == 0:
                l_bounds, inner_sums, sizes = start_elkan(start_dists, kernel_matrix, labels, self.n_clusters)
            else:
                l_bounds, inner_sums, sizes, center_dists = update_elkan(kernel_matrix, l_bounds, center_dists, labels, labels_old, sizes, inner_sums, self.n_clusters)

            labels_old = labels
            labels = np.array(np.argmin(l_bounds, axis=1), dtype=np.int_)
            
            quality, converged = self.measure_iter(l_bounds, labels, labels_old, quality)
            self._out_verbose(it, quality, converged=converged)
            if converged:
                break

        return labels, quality, inner_sums, sizes





    



            

            
            

            
            
            
