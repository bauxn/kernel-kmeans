# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange

cdef extern from "float.h":
    cdef double DBL_MAX

ctypedef fused llong:
    long
    long long


def calc_silhouettes(double[:, ::1] distances, llong[::1] labels):
    cdef:
        Py_ssize_t data_size = distances.shape[0]
        Py_ssize_t n_clusters = distances.shape[1]
        double[::1] silhouettes = np.zeros(data_size)
        Py_ssize_t i, j
        llong labels_i
        double a, b

    for i in prange(data_size, nogil=True):
        labels_i = labels[i]
        a = distances[i, labels_i]
        b = DBL_MAX
        for j in range(n_clusters):
            if j == labels_i:
                continue
            b = min(distances[i, j], b)

        silhouettes[i] = (b - a) / max(a,b) 
    
    return np.asarray(silhouettes) 


cdef double max(double a, double b) nogil:
    if a > b:
        return a
    return b

cdef double min(double a, double b) nogil:
    if a < b:
        return a
    return b


'''
def calc_sq_distances_full(double[:, ::1] kernel_matrix):
    cdef:
        Py_ssize_t rows = kernel_matrix.shape[0]
        Py_ssize_t cols = kernel_matrix.shape[1]
        Py_ssize_t i, j
        double kernel_ii
        double[:, ::1] dists = np.zeros((rows, cols))
    
    for i in prange(rows, nogil=True):
        kernel_ii = kernel_matrix[i, i]
        for j in range(cols):
            dists[i, j] = kernel_ii - 2*kernel_matrix[i, j] + kernel_matrix[j, j]
    
    return np.asarray(dists)
'''

