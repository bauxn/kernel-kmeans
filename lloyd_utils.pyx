# cython: boundscheck=False
# cython: wraparound=False
# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange


# Best way to deal with varying datatypes
# MUST NOT BE CONST as cython compiler breaks otherwise
ctypedef fused llong:
    long
    long long


def lloyd_update(distances, 
                 const double[:, ::1] kernel_matrix,
                 llong[::1] labels, 
                 const long n_clusters):

    outer_sums, inner_sums, cluster_sizes =\
        _calc_update(kernel_matrix, labels, n_clusters)
    
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster] 
        distances[:,  cluster] += (-2 * outer_sum / size + 
                                  inner_sum / size**2)
    return distances, inner_sums, cluster_sizes

    

def _calc_update(const double[:, ::1] kernel_matrix, llong[::1] labels, const long n_clusters):

    cdef:
        int size = kernel_matrix.shape[0]
        double[:, ::1] inner_sum = np.zeros((size, n_clusters), dtype=np.float64)
        double[:, ::1] outer_sum = np.zeros((size, n_clusters), dtype=np.float64)
        int[:, ::1] cluster_sizes = np.zeros((size, n_clusters), dtype=np.int32)
        int i,j
        double kernel_ij
        int label_i, label_j
    for i in prange(size, nogil=True):
        label_i = labels[i]
        cluster_sizes[i, label_i] += 1
        for j in range(size):
            label_j = labels[j]
            kernel_ij = kernel_matrix[i, j]
            outer_sum[i, label_j] += kernel_ij      
            if label_i == label_j:
                inner_sum[i, label_j] += kernel_ij
    return np.asarray(outer_sum), np.sum(inner_sum, axis=0), np.sum(cluster_sizes, axis = 0)



def calc_distances(inner_sums,
                  cluster_sizes,
                  const double[:, ::1] kernel_matrix,
                  llong[::1] labels,
                  const long n_clusters):
    outer_sums = np.array(_calc_outer_sums(kernel_matrix, labels, n_clusters))
    distances = np.zeros((kernel_matrix.shape[0], n_clusters))
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster]
        distances[:, cluster] += (-2 * outer_sum / size +
                                 inner_sum / size**2)
    return distances


def _calc_outer_sums(const double[:, ::1] kernel_matrix, llong[::1] labels, const long n_clusters):
    cdef:
        int rows = kernel_matrix.shape[0]
        int cols = kernel_matrix.shape[1]
        double[:, ::1] outer_sums = np.zeros((rows, n_clusters), dtype=np.float64)
        int i,j
    for i in prange(rows, nogil=True):
        for j in range(cols):
            outer_sums[i, labels[j]] += kernel_matrix[i, j]    
    return outer_sums
