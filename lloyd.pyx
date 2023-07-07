import numpy as np
from cython.parallel import prange
from utils import fill_empty_clusters

def update_lloyd(sq_distances, 
                 const double[:, ::1] kernel_matrix,
                 long[::1] labels, 
                 const long n_clusters):
                 
    cdef Py_ssize_t cluster

    labels, cluster_sizes = fill_empty_clusters(labels, n_clusters, return_sizes=True)
    outer_sums, inner_sums = calc_sums_full(kernel_matrix, labels, n_clusters)
    
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster] 
        sq_distances[:,  cluster] += (-2 * outer_sum / size + 
                                  inner_sum / size**2)
    return sq_distances, inner_sums, cluster_sizes


def calc_sums_full(const double[:, ::1] kernel_matrix, long[::1] labels, const long n_clusters):

    cdef:
        int rows = kernel_matrix.shape[0]
        int cols = kernel_matrix.shape[1]
        double[:, ::1] inner_sum = np.zeros((rows, n_clusters), dtype=np.double)
        double[:, ::1] outer_sum = np.zeros((rows, n_clusters), dtype=np.double)
        int i,j
        int label_i, label_j
    for i in prange(rows, nogil=True):
        label_i = labels[i]
        for j in range(cols):
            label_j = labels[j]
            outer_sum[i, label_j] += kernel_matrix[i, j]      
            if label_i == label_j:
                inner_sum[i, label_j] += kernel_matrix[i, j]
    return np.asarray(outer_sum), np.sum(inner_sum, axis=0)
