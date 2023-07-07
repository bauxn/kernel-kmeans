import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt 
from utils import fill_empty_clusters



def update_elkan(
        double[:, ::1] kernel_matrix, 
        double[:, ::1] l_bounds, 
        double[:, ::1] center_dists, 
        long[::1] labels, 
        long[::1] labels_old, 
        long[::1] sizes, 
        double[::1] inner_sums,
        long n_clusters):
    sizes_old = sizes
    labels, sizes = fill_empty_clusters(labels, n_clusters)
    inner_sums_old = inner_sums
    inner_sums_mixed, inner_sums = calc_inner_sums_mixed(kernel_matrix, labels, labels_old, n_clusters, return_new_sums=True)
    center_dists += calc_center_dists(inner_sums, inner_sums_mixed, inner_sums_old, sizes, sizes_old)
    l_bounds = est_lower_bounds(kernel_matrix, l_bounds, center_dists, labels, sizes, inner_sums)

    return np.asarray(l_bounds), inner_sums, sizes, center_dists


def est_lower_bounds(
        double[:, ::1] kernel_matrix, 
        double[:, ::1] l_bounds, 
        double[:, ::1] center_dists, 
        long[::1] labels, 
        long[::1] sizes, 
        double[::1] inner_sums):
    cdef:
        Py_ssize_t i, j, labels_i
        double outer_sum

    assert l_bounds.shape[0] == labels.shape[0]
    assert l_bounds.shape[1] == inner_sums.shape[0]
    

    for i in prange(l_bounds.shape[0], nogil=True):
        labels_i = labels[i]
        outer_sum = calc_outer_sum_single(kernel_matrix, i, labels_i, labels)
        l_bounds[i, labels_i] = (
            kernel_matrix[i, i] 
            - 2 * outer_sum / sizes[labels_i] 
            + inner_sums[labels_i] / sizes[labels_i]**2
        )
        center_dists[i, labels_i] = 0
        for j in range(l_bounds.shape[1]):
            if sqrt(l_bounds[i, j]) - center_dists[i, j] < sqrt(l_bounds[i, labels_i]): 
                outer_sum = calc_outer_sum_single(kernel_matrix, i, j, labels)
                l_bounds[i, j] = kernel_matrix[i, i] - 2 * outer_sum / sizes[j] + inner_sums[j] / sizes[j]**2
                center_dists[i, j] = 0
    return np.asarray(l_bounds)


def calc_center_dists(double[::1] inner_sums_new, double[::1] inner_sums_mixed, double[::1] inner_sums_old, long[::1] sizes_new, long[::1] sizes_old):
    cdef:
        Py_ssize_t i
        Py_ssize_t n_clusters = inner_sums_new.shape[0]
        double[::1] dists = np.zeros(n_clusters)
        long new_size, old_size
        double mixed_sum, new_sum, old_sum

    #TODO not assert here (raise exception)
    assert n_clusters == inner_sums_mixed.shape[0] 
    assert n_clusters == inner_sums_old.shape[0]
    assert n_clusters == sizes_new.shape[0]
    assert n_clusters == sizes_old.shape[0]

    for i in range(n_clusters):
        new_size = sizes_new[i]
        old_size = sizes_old[i]
        mixed_sum = inner_sums_mixed[i]
        new_sum = inner_sums_new[i]
        old_sum = inner_sums_old[i]
        dists[i] = sqrt(new_sum / new_size**2 - 2 * mixed_sum / (new_size * old_size) + old_sum / old_size**2)
    
    return np.asarray(dists) 
    



def calc_inner_sums_mixed(double[:, ::1] kernel_matrix, long[::1] labels, long[::1] labels_old, long n_clusters, return_new_sums=True):
    cdef:
        Py_ssize_t rows = kernel_matrix.shape[0]
        Py_ssize_t cols = kernel_matrix.shape[1]
        double[:, ::1] sums_mixed = np.zeros((rows, n_clusters))
        double[:, ::1] sums_new = np.zeros((rows, n_clusters))
        Py_ssize_t i, j
        long labels_i, labels_j_new, labels_j_old
    
    for i in prange(rows, nogil=True):
        labels_i = labels[i]
        for j in range(cols):
            labels_j_new = labels[j] 
            labels_j_old = labels_old[j] 
            if labels_i == labels_j_new:
                sums_new[i, labels_j_new] += kernel_matrix[i, j]
            if labels_i == labels_j_old:
                sums_mixed[i, labels_j_old] += kernel_matrix[i, j]
    if return_new_sums:
        return np.sum(sums_mixed, axis=0), np.sum(sums_new, axis=0)
    return np.sum(sums_mixed, axis=0)



# TODO start_elkan == update_lloyd, added here to not break either if there are changes
def start_elkan(sq_distances, 
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


cdef double calc_outer_sum_single(double[:, ::1] kernel_matrix, long elem_index, long cluster_index, long[::1] labels) nogil:
    cdef:
        double outer_sum = 0.
        Py_ssize_t i
        Py_ssize_t size = len(labels)
    for i in range(size):
        if labels[i] == cluster_index:
            outer_sum += kernel_matrix[elem_index, i]
    return outer_sum
