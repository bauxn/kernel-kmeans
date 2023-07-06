import numpy as np
from cython.parallel import prange



def fill_empty_clusters(long[::1] labels, const long n_clusters, return_sizes=True):
    cdef Py_ssize_t i, index
    cdef long size = labels.size
    
    rng = np.random.default_rng(0)

    while True:
        cluster_sizes = np.asarray(calc_sizes(labels, n_clusters), dtype=np.int_)
        empty_cluster_indices = np.flatnonzero(cluster_sizes == 0)
        amount_empty_clusters = empty_cluster_indices.size
        if(amount_empty_clusters == 0):
            break
        for i in range(amount_empty_clusters):
            print("Warning! Empty cluster encountered, consider using different n_cluster. Random element assigned to emtpy cluster")
            index = rng.integers(size)
            labels[index] = empty_cluster_indices[i]
    if return_sizes:
        return np.asarray(labels, dtype=np.int_), np.asarray(cluster_sizes, dtype=np.int_)
    return np.asarray(labels, dtype=np.int_)

cpdef long[::1] calc_sizes(long[::1] labels, const long n_clusters):
    cdef:
        Py_ssize_t size = labels.shape[0] 
        Py_ssize_t i
        long[::1] sizes = np.zeros(n_clusters, dtype=np.int_)
    for i in range(size):
        sizes[labels[i]] += 1
    return sizes


def calc_sq_distances(inner_sums,
                  cluster_sizes,
                  const double[:, ::1] kernel_matrix,
                  long[::1] labels,
                  const long n_clusters):
    outer_sum_full = np.array(_calc_outer_sum_full(kernel_matrix, labels, n_clusters))
    distances = np.tile(np.diag(kernel_matrix), (n_clusters, 1)).T
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sum_full[:, cluster]
        inner_sum = inner_sums[cluster]
        distances[:, cluster] += (-2 * outer_sum / size +
                                 inner_sum / size**2)
    return distances



cpdef double[:, ::1] _calc_outer_sum_full(const double[:, ::1] kernel_matrix, long[::1] labels, const long n_clusters):
    cdef:
        int rows = kernel_matrix.shape[0]
        int cols = kernel_matrix.shape[1]
        double[:, ::1] outer_sums = np.zeros((rows, n_clusters), dtype=np.float64)
        int i,j
    for i in prange(rows, nogil=True):
        for j in range(cols):
            outer_sums[i, labels[j]] += kernel_matrix[i, j]    
    return outer_sums

cpdef double calc_outer_sum_single(double[:, ::1] kernel_matrix, long elem_index, long cluster_index, long[::1] labels) nogil:
    cdef:
        double outer_sum = 0.
        Py_ssize_t i
        Py_ssize_t size = len(labels)
    for i in range(size):
        if labels[i] == cluster_index:
            outer_sum += kernel_matrix[elem_index, i]
    return outer_sum


def calc_sums_full(const double[:, ::1] kernel_matrix, long[::1] labels, const long n_clusters):

    cdef:
        int size = kernel_matrix.shape[0]
        double[:, ::1] inner_sum = np.zeros((size, n_clusters), dtype=np.double)
        double[:, ::1] outer_sum = np.zeros((size, n_clusters), dtype=np.double)
        int i,j
        double kernel_ij
        int label_i, label_j
    for i in prange(size, nogil=True):
        label_i = labels[i]
        for j in range(size):
            label_j = labels[j]
            kernel_ij = kernel_matrix[i, j]
            outer_sum[i, label_j] += kernel_ij      
            if label_i == label_j:
                inner_sum[i, label_j] += kernel_ij
    return np.asarray(outer_sum), np.sum(inner_sum, axis=0)
