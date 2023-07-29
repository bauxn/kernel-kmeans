import numpy as np
from cython.parallel import prange

def fill_empty_clusters(long[::1] labels, long n_clusters, return_sizes=True, rng=None):
    '''
    Utility to fill empty clusters

    Calculates cluster sizes via labels, if any size is zero
    a random element is assigned to empty cluster.
    EG cluster 0 has size 0, labels[random_index] = 0.
    At the beginning a new rng-generator is created to ensure
    predictable assignments. 
    As sizes must be calculated, can be used as alternative to
    calc_sizes

    Parameters
    ----------
    labels: ndarray of shape(n_samples), dtype=np.int_
    n_clusters: int
    return_sizes: bool
    rng: int
        seed used for random number generator

    Returns
    -------
    labels: ndarray of shape(n_samples), dtype=np.int_
        corrected labels so that no cluster is empty.
        if this has been fulfilled at the beginning,
        nothing changes.
    cluster_sizes: ndarray of shape(n_clusters), dtype=int_, optional
    '''

    cdef Py_ssize_t i, index
    cdef long n_samples = labels.size
    
    if n_samples < n_clusters:
        raise ValueError("n_samples must be > n_clusters")

    rng = np.random.default_rng(rng)

    while True:
        cluster_sizes = np.asarray(calc_sizes(labels, n_clusters), dtype=np.int_)
        # equal to: indices=[c for c in range(n_clusters) if cluster_sizes[c]==0]
        empty_cluster_indices = np.flatnonzero(cluster_sizes == 0)
        amount_empty_clusters = empty_cluster_indices.size
        if(amount_empty_clusters == 0):
            break
        for i in range(amount_empty_clusters):
            #assign every empty cluster one element
            print("Warning! Empty cluster encountered, consider using different n_cluster. Random element assigned to emtpy cluster")
            index = rng.integers(n_samples)
            labels[index] = empty_cluster_indices[i]
    if return_sizes:
        return np.asarray(labels, dtype=np.int_), np.asarray(cluster_sizes, dtype=np.int_)
    return np.asarray(labels, dtype=np.int_)


cpdef long[::1] calc_sizes(long[::1] labels, long n_clusters):
    '''calculates the size of every cluster'''
    cdef:
        Py_ssize_t size = labels.shape[0] 
        Py_ssize_t i
        long[::1] sizes = np.zeros(n_clusters, dtype=np.int_)
    for i in range(size):
        sizes[labels[i]] += 1
    return sizes


def calc_sq_distances(
        inner_sums,
        cluster_sizes,
        double[:, ::1] kernel_matrix,
        long[::1] labels,
        long n_clusters):
    '''
    computes sq_dists form every point to every cluster

    Intended to be used for predictions, as inner_sums, 
    cluster_sizes and labels are not extra computed here.

    Parameters
    ----------
    inner_sums: ndarray of shape(n_clusters), dtype=np.double
        inner_sums[i] = sum(K(x,y)) for all x, y in cluster i
    cluster_sizes: ndarray of shape(n_clusters), dtype=np.int_
    kernel_matrix: ndarray of shape(n_samples, n_pred_samples), dtype=np.double
        kernelmatrix between the fitted data and the samples to predict
    labels: ndarray of shape(n_samples), dtype=np.int_
    n_clusters: int

    Returns
    -------
    distances: ndarray of shape(n_pred_samples, n_clusters)
        sq distances of each sample to predict to each center
    '''
    outer_sum_full = np.asarray(_calc_outer_sums(kernel_matrix, labels, n_clusters))
    sq_distances = np.zeros((kernel_matrix.shape[0], n_clusters))
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sum_full[:, cluster]
        inner_sum = inner_sums[cluster]
        sq_distances[:, cluster] += (-2 * outer_sum / size 
                                    + inner_sum / size**2)
    return np.asarray(sq_distances)


cpdef double[:, ::1] _calc_outer_sums(
        double[:, ::1] kernel_matrix, 
        long[::1] labels, 
        long n_clusters):
    '''
    Mathematical helper function
    
    outer_sums[i, j] == sum(K(data[i], data[x])) for x in cluster j
    In words, the sum of the kernel of datapoint X[i] and all datapoints
    in cluster number j.
    '''
    cdef:
        Py_ssize_t rows = kernel_matrix.shape[0]
        Py_ssize_t cols = kernel_matrix.shape[1]
        double[:, ::1] outer_sums = np.zeros((rows, n_clusters), dtype=np.double)
        Py_ssize_t i,j
    for i in prange(rows, nogil=True):
        for j in range(cols):
            outer_sums[i, labels[j]] += kernel_matrix[i, j]    
    return outer_sums

cpdef double[::1] _calc_inner_sums(
        double[:, ::1] kernel_matrix, 
        long[::1] labels, 
        long n_clusters):
    '''
    just used for measurements
    '''
    cdef:
        Py_ssize_t rows = kernel_matrix.shape[0]
        Py_ssize_t cols = kernel_matrix.shape[1]
        double[:, ::1] inner_sums = np.zeros(shape=(rows, n_clusters), dtype=np.double)
        Py_ssize_t i,j
        long labels_i
    for i in prange(rows, nogil=True):
        labels_i = labels[i]
        for j in range(cols):
            if labels[j] == labels_i:
                inner_sums[i, labels_i] += kernel_matrix[i, j]    
    return np.sum(inner_sums, axis=0)
