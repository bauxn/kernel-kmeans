# cython: boundscheck=False
# cython: wraparound=False
# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange
from utils import fill_empty_clusters

def update_lloyd(sq_distances, 
                 const double[:, ::1] kernel_matrix,
                 long[::1] labels, 
                 const long n_clusters):
                 
    cdef Py_ssize_t cluster

    labels, cluster_sizes = fill_empty_clusters(labels, n_clusters, return_sizes=True)
    outer_sums, inner_sums = _calc_update(kernel_matrix, labels, n_clusters)
    
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster] 
        sq_distances[:,  cluster] += (-2 * outer_sum / size + 
                                  inner_sum / size**2)
    return sq_distances, inner_sums, cluster_sizes


def _calc_update(const double[:, ::1] kernel_matrix, long[::1] labels, const long n_clusters):

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
