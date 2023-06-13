# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = /openmp

import numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs



def update_elkan(
        double[:, ::1] kernel_matrix, 
        double[:, ::1] l_bounds, 
        double[:, ::1] center_dists, 
        long[::1] labels, 
        long[::1] labels_old, 
        long[::1] sizes, 
        double[::1] inner_sums,
        long n_clusters
    ):

    sizes_old = sizes
    sizes = np.array(calc_sizes(labels, n_clusters), dtype=np.int_)
    inner_sums_old = inner_sums
    inner_sums, inner_sums_mixed = calc_inner_sums(kernel_matrix, labels, labels_old, n_clusters)
    center_dists += calc_center_dists(inner_sums, inner_sums_mixed, inner_sums_old, sizes, sizes_old)
    l_bounds = est_lower_bounds(kernel_matrix, l_bounds, center_dists, labels, sizes, inner_sums)

    return np.asarray(l_bounds), inner_sums, sizes, center_dists


def est_lower_bounds(
        double[:, ::1] kernel_matrix, 
        double[:, ::1] l_bounds, 
        double[:, ::1] center_dists, 
        long[::1] labels, 
        long[::1] sizes, 
        double[::1] inner_sums
    ):

    cdef:
        Py_ssize_t i, j, labels_i
        double outer_sum

    assert l_bounds.shape[0] == labels.shape[0]
    assert l_bounds.shape[1] == inner_sums.shape[0]
    

    for i in prange(l_bounds.shape[0], nogil=True):
        labels_i = labels[i]
        outer_sum = calc_outer_sum(kernel_matrix, i, labels_i, labels)
        l_bounds[i, labels_i] = (kernel_matrix[i, i] 
            - 2 * outer_sum / sizes[labels_i] 
            + inner_sums[labels_i] / sizes[labels_i]**2
        )
        center_dists[i, labels_i] = 0
        for j in range(l_bounds.shape[1]):
            if fabs(sqrt(l_bounds[i, j]) - center_dists[i, j]) < sqrt(l_bounds[i, labels_i]): #or j == labels_i:
                outer_sum = calc_outer_sum(kernel_matrix, i, j, labels)
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
    
    return np.array(dists) 
    



def calc_inner_sums(double[:, ::1] kernel_matrix, long[::1] labels, long[::1] labels_old, long n_clusters):
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
    
    return np.sum(sums_new, axis=0), np.sum(sums_mixed, axis=0)

cpdef double calc_outer_sum(double[:, ::1] kernel_matrix, long elem_index, long cluster_index, long[::1] labels) nogil:
    cdef:
        double outer_sum = 0.
        Py_ssize_t i
        Py_ssize_t size = len(labels)
    for i in range(size):
        if labels[i] == cluster_index:
            outer_sum += kernel_matrix[elem_index, i]
    return outer_sum

cpdef long[::1] calc_sizes(long[::1] labels, long n_clusters):
    cdef:
        Py_ssize_t size = labels.shape[0] 
        Py_ssize_t i
        long[::1] sizes = np.zeros(n_clusters, dtype=np.int_)
    for i in range(size):
        sizes[labels[i]] += 1
    return sizes


def est_lower_bounds_old(
        double[:, ::1] kernel_matrix, 
        double[:, ::1] l_bounds, 
        double[:, ::1] center_dists, 
        long[::1] labels, 
        long[::1] sizes, 
        double[::1] inner_sums
    ):

    cdef:
        Py_ssize_t i, j, labels_i
        double outer_sum

    assert l_bounds.shape[0] == labels.shape[0]
    assert l_bounds.shape[1] == inner_sums.shape[0]
    

    for i in prange(l_bounds.shape[0], nogil=True):
        labels_i = labels[i]
        for j in range(l_bounds.shape[1]):
            if fabs(sqrt(l_bounds[i, j]) - center_dists[i, j]) < sqrt(l_bounds[i, labels_i]):
                outer_sum = calc_outer_sum(kernel_matrix, i, j, labels)
                l_bounds[i, j] = kernel_matrix[i, i] - 2 * outer_sum / sizes[j] + inner_sums[j] / sizes[j]**2
                center_dists[i, j] = 0
    return np.asarray(l_bounds)



# def est_lower_bounds_VERY_old(double[:, ::1] kernel_matrix, double[:, ::1] l_bounds, double[::1] center_dists, long[::1] labels, long[::1] sizes, double[::1] inner_sums):
#     
#     cdef:
#         Py_ssize_t i, j, labels_i
#         double outer_sum
    
#     for i in prange(l_bounds.shape[0], nogil=True):
#         labels_i = labels[i]
#         for j in range(l_bounds.shape[1]):
            
#             if fabs(sqrt(l_bounds[i, j]) - center_dists[j]) < sqrt(l_bounds[i, labels_i]) or j == labels_i:
#                 outer_sum = calc_outer_sum(kernel_matrix, i, j, labels)
#                 l_bounds[i, j] = kernel_matrix[i, i] - 2 * outer_sum / sizes[j] + inner_sums[j] / sizes[j]**2

#     return np.asarray(l_bounds)