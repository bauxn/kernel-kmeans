# cython: boundscheck=False
# cython: wraparound=False
# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange
from libc.stdlib cimport rand


def lloyd_update(sq_distances, 
                 const double[:, ::1] kernel_matrix,
                 long[::1] labels, 
                 const long n_clusters):
                 
    cdef Py_ssize_t cluster

    labels, cluster_sizes = _fill_empty_clusters(labels, n_clusters)
    outer_sums, inner_sums = _calc_update(kernel_matrix, labels, n_clusters)
    
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster] 
        sq_distances[:,  cluster] += (-2 * outer_sum / size + 
                                  inner_sum / size**2)
    return sq_distances, inner_sums, cluster_sizes

def _fill_empty_clusters(long[::1] labels, const long n_clusters):
    cdef Py_ssize_t i, index
    cdef long l_size = labels.size
    
    while True:
        cluster_sizes = np.asarray(calc_sizes(labels, n_clusters))
        empty_cluster_indices = np.flatnonzero(cluster_sizes == 0)
        amount_empty_clusters = empty_cluster_indices.size
        if(amount_empty_clusters == 0):
            break
        for i in range(amount_empty_clusters):
            print("Warning! Empty cluster encountered, consider using different n_cluster. Random element assigned to emtpy cluster")
            index = rand() % l_size
            labels[index] = empty_cluster_indices[i]
    return labels, np.asarray(cluster_sizes)
    
cpdef calc_sizes(long[::1] labels, const long n_clusters):
    cdef:
        Py_ssize_t size = labels.shape[0] 
        Py_ssize_t i
        int[::1] sizes = np.zeros(n_clusters, dtype=np.int32)
    for i in range(size):
        sizes[labels[i]] += 1
    return sizes


def _calc_update(const double[:, ::1] kernel_matrix, long[::1] labels, const long n_clusters):

    cdef:
        int size = kernel_matrix.shape[0]
        double[:, ::1] inner_sum = np.zeros((size, n_clusters), dtype=np.float64)
        double[:, ::1] outer_sum = np.zeros((size, n_clusters), dtype=np.float64)
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


def calc_sq_distances(inner_sums,
                  cluster_sizes,
                  const double[:, ::1] kernel_matrix,
                  long[::1] labels,
                  const long n_clusters):
    outer_sums = np.array(_calc_outer_sums(kernel_matrix, labels, n_clusters))
    distances = np.tile(np.diag(kernel_matrix), (n_clusters, 1)).T
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster]
        distances[:, cluster] += (-2 * outer_sum / size +
                                 inner_sum / size**2)
    return distances


def _calc_outer_sums(const double[:, ::1] kernel_matrix, long[::1] labels, const long n_clusters):
    cdef:
        int rows = kernel_matrix.shape[0]
        int cols = kernel_matrix.shape[1]
        double[:, ::1] outer_sums = np.zeros((rows, n_clusters), dtype=np.float64)
        int i,j
    for i in prange(rows, nogil=True):
        for j in range(cols):
            outer_sums[i, labels[j]] += kernel_matrix[i, j]    
    return outer_sums
