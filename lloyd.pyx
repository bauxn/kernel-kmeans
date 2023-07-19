import numpy as np
from cython.parallel import prange
from utils import fill_empty_clusters

def update_lloyd(
        sq_distances, 
        double[:, ::1] kernel_matrix,
        long[::1] labels, 
        long n_clusters):
    '''
    Computes squared dists from points to centers in feature space
 
    First, checks for empty cluster and fills them with a random 
    element. Then computes the sq_dist from each point to each center.
    
    Parameters
    ----------
    sq_distances: ndarray of shape(n_samples, n_clusters), dtype=np.double
        sq_distances[x, :] = K(x,x). Could be zeros but then inertia
        quality measure would not be exact.
    kernel_matrix: ndarray of shape(n_samples, n_samples), dtype=double
    labels: ndarray of shape(n_samples), dtype=long
    n_clusters: long

    Returns
    -------
    sq_distances: ndarray of shape(n_samples, n_clusters)
    inner_sums: ndarray of shape(n_clusters)
        sum of pairwise sq_dists in each cluster, used
        for prediction
    cluster_sizes: ndarray of shape(n_clusters)
    
    Notes
    -----
    Numpy ndarray can be passed for memoryview.
    sq_distances parameter deliberately not typed because
    of np broadcast addition in loop.
    TODO measure speed if typed explicitely.
    '''         
    cdef Py_ssize_t cluster # loop index

    sq_distances = np.asarray(sq_distances, dtype=np.double)

    labels, cluster_sizes = fill_empty_clusters(labels, n_clusters, return_sizes=True)
    outer_sums, inner_sums = _calc_sums_full(kernel_matrix, labels, n_clusters)
    
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster] 
        sq_distances[:,  cluster] += (-2 * outer_sum / size 
                                     + inner_sum / size**2)
    return np.asarray(sq_distances), np.asarray(inner_sums), np.asarray(cluster_sizes)


def _calc_sums_full(double[:, ::1] kernel_matrix, long[::1] labels, long n_clusters):
    '''
    Helper function for distance calculation.

    Parameters
    ----------
    kernel_matrix: ndarray of shape(n_samples, n_samples), dtype=double
        pairwise kernel matrix of input data
    labels: ndarray of shape(n_samples), dtype=long
    n_clusters: int

    Returns
    -------
    outer_sums: ndarray of shape(n_samples, n_clusters)
        outer_sums[i, j] == sum(K(data[i], data[x])) for x in cluster j
        In words, the sum of the kernel of datapoint X[i] and all datapoints
        in cluster number j.
    inner_sums: ndarray of shape(n_clusters)
        "inner 'distance' sum"; 
        inner_sums[i] = sum(K(x,y)) for all x, y in cluster i
    '''


    cdef:
        Py_ssize_t size = kernel_matrix.shape[0]
        double[:, ::1] inner_sum = np.zeros((size, n_clusters), dtype=np.double)
        double[:, ::1] outer_sum = np.zeros((size, n_clusters), dtype=np.double)
        Py_ssize_t i, j
        int label_j

    assert size == kernel_matrix.shape[1]

    for i in prange(size, nogil=True):
        for j in range(size):
            label_j = labels[j]
            outer_sum[i, label_j] += kernel_matrix[i, j]      
            if labels[i] == label_j:
                inner_sum[i, label_j] += kernel_matrix[i, j]
    return np.asarray(outer_sum), np.sum(inner_sum, axis=0)