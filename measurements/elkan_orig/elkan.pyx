# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = /openmp

import numpy as np
from cython.parallel cimport prange
from libc.math cimport sqrt
from time import time

def update_elkan(
        double[:, ::1] kmatrix,
        long[::1] labels,
        double[:, ::1] lbounds,
        double[::1] ubounds,
        long n_clusters):
    cdef:
        Py_ssize_t size = kmatrix.shape[0]
        double[:, ::1] p_cdists = np.zeros((n_clusters, n_clusters))
        long[::1] labels_new = np.copy(labels)
        double[::1] s = np.zeros(n_clusters)
        long[::1] cl_sizes = np.zeros(n_clusters, dtype=np.int_)
        double[::1] inner_sums = np.zeros(n_clusters)
        double[:, ::1] mixed_sums = np.zeros((size, n_clusters))
        Py_ssize_t x, c
        long c_x
        bint r

    assert size == kmatrix.shape[1]
    assert size == labels.shape[0]
    assert size == lbounds.shape[0]
    assert n_clusters == lbounds.shape[1]
    assert size == ubounds.shape[0]


    cl_sizes = calc_sizes(labels, n_clusters)
    start = time()
    inner_sums, mixed_sums = calc_pairwise_mixed_sums(kmatrix, labels, n_clusters)
    end = time()
    print("pairwise:", end - start)
    p_cdists = calc_pairwise_centerdists(inner_sums, mixed_sums, cl_sizes, n_clusters)
    s = calc_s(p_cdists, n_clusters)


    for x in prange(size, nogil=True):
        r = True
        if ubounds[x] <= s[labels_new[x]]:
            continue
        for c in range(n_clusters):
            c_x = labels_new[x]
            if c == c_x or ubounds[x] <= lbounds[x, c] or ubounds[x] <= p_cdists[c, c_x] / 2:
                continue

            if r:
                ubounds[x] = sqrt(calc_sq_point_cluster_dist(kmatrix, x, c_x, inner_sums, labels, cl_sizes))
                r = False

            if ubounds[x] > lbounds[x, c] or ubounds[x] > p_cdists[c, c_x] / 2:
                lbounds[x, c] = sqrt(calc_sq_point_cluster_dist(kmatrix, x, c, inner_sums, labels, cl_sizes))
                if lbounds[x, c] < ubounds[x]:
                    labels_new[x] = c
                    ubounds[x] = lbounds[x, c]
    
    return np.asarray(lbounds), np.asarray(ubounds), np.asarray(labels_new)




cdef double calc_sq_point_cluster_dist(
        double[:, ::1] kmatrix,
        long x,
        long c, 
        double[::1] inner_sums,
        long[::1] labels,
        long[::1] sizes) nogil:

    return (kmatrix[x, x]
            - 2 * calc_outer_sum_single(kmatrix, x, c, labels) / sizes[c]
            + inner_sums[c] / sizes[c]**2)



cdef double calc_outer_sum_single(
        double[:, ::1] kmatrix,
        long x,
        long c,
        long[::1] labels) nogil:
    cdef:
        Py_ssize_t size = kmatrix.shape[0]
        double outer = 0.
        Py_ssize_t y
    
    for y in range(size):
        if labels[y] == c:
            outer += kmatrix[x, y]

    return outer
    
def calc_s(double[:, ::1] p_cdists, long n_clusters):
    cdef:
        Py_ssize_t c
        double[::1] s = np.zeros(n_clusters)
    
    for c in range(n_clusters):
        s[c] = np.partition(p_cdists[c], 1)[1]/2 
    return s

def calc_pairwise_centerdists(
        double[::1] inner_sums,
        double[:, ::1] mixed_sums,
        long[::1] sizes, 
        long n_clusters):
    cdef:
        Py_ssize_t c, c_
        double[:, ::1] p_cdists = np.zeros((n_clusters, n_clusters))
        double cdist
    
    for c in prange(n_clusters, nogil=True):
        for c_ in range(n_clusters):
            cdist = (inner_sums[c] / sizes[c]**2
                    - 2 * mixed_sums[c, c_] / (sizes[c] * sizes[c_])
                    + inner_sums[c_] / sizes[c_]**2)
            p_cdists[c, c_] = cdist
            
    return np.sqrt(p_cdists)
    



def calc_pairwise_mixed_sums(
        double[:, ::1] kmatrix, 
        long[::1] labels, 
        long n_clusters):
    cdef:
        Py_ssize_t rows = kmatrix.shape[0]
        Py_ssize_t cols = kmatrix.shape[1]
        double[:, :, ::1] sums_mixed = np.zeros((rows, n_clusters, n_clusters))
        double[:, ::1] sums_new = np.zeros((rows, n_clusters))
        Py_ssize_t x, y
        long cl_x, cl_y
    
    assert rows == cols
    assert cols == labels.shape[0]

    for x in prange(rows, nogil=True):
        cl_x = labels[x]
        for y in range(cols):
            cl_y = labels[y]
            sums_mixed[x, cl_x, cl_y] += kmatrix[x, y]
            if cl_x == cl_y:
                sums_new[x, cl_x] += kmatrix[x, y]
    
    return np.sum(sums_new, axis=0), np.sum(sums_mixed, axis=0)



def init_lbounds(
        double[:, ::1] kmatrix, 
        long[::1] labels, 
        long n_clusters):    
    cdef:
        Py_ssize_t size = labels.shape[0]
        double[:, ::1] dists = np.zeros((size, n_clusters))
        double[:, ::1] outer_sums = np.zeros((size, n_clusters))
        double[::1] inner_sums = np.zeros(size)
        long[::1] sizes = np.zeros(size, dtype=np.int_)
        Py_ssize_t x, c
    dists = np.ascontiguousarray(np.tile(np.diag(kmatrix), (n_clusters, 1)).T)
    sizes = calc_sizes(labels, n_clusters)
    outer_sums, inner_sums = calc_sums_full(kmatrix, labels, n_clusters)
    for x in prange(size, nogil=True):
        for c in range(n_clusters):
            dists[x, c] += inner_sums[c]/sizes[c] ** 2 - 2 * outer_sums[x, c] / sizes[c]
    
    return np.sqrt(dists)

def calc_center_movement(
        double[:, ::1] kmatrix,
        long[::1] labels_new,
        long[::1] labels_old,
        long n_clusters):
    cdef:
        double[::1] c_dists = np.zeros(n_clusters)
        long[::1] sizes_new = np.zeros(n_clusters, dtype=np.int_)
        long[::1] sizes_old = np.zeros(n_clusters, dtype=np.int_)
        double[::1] inner_sums_new, mixed_sums, inner_sums_old
        Py_ssize_t c
    
    assert kmatrix.shape[0] == kmatrix.shape[1]
    assert kmatrix.shape[0] == labels_new.shape[0]
    assert kmatrix.shape[0] == labels_old.shape[0]
    
    sizes_new = calc_sizes(labels_new, n_clusters)
    sizes_old = calc_sizes(labels_old, n_clusters)
    inner_sums_new, mixed_sums, inner_sums_old = calc_mixed_sums(kmatrix, labels_new, labels_old, n_clusters)
    
    for c in range(n_clusters):
        c_dists[c] += (inner_sums_new[c] / sizes_new[c]**2 
                    - 2 * mixed_sums[c] / (sizes_new[c] * sizes_old[c]) 
                    + inner_sums_old[c] / sizes_old[c]**2)
    
    return np.sqrt(c_dists, dtype=np.double, order="C")

def calc_mixed_sums(
        double[:, ::1] kmatrix,
        long[::1] labels_new,
        long[::1] labels_old,
        long n_clusters):
    cdef:
        Py_ssize_t rows = kmatrix.shape[0]
        Py_ssize_t cols = kmatrix.shape[1]
        double[:, ::1] inner_new = np.zeros((rows, n_clusters))
        double[:, ::1] inner_old = np.zeros((rows, n_clusters))
        double[:, ::1] mixed = np.zeros((rows, n_clusters))
        Py_ssize_t x, y
        long c_x, c_y_new, c_y_old

    assert rows == cols
    assert cols == labels_new.shape[0]
    assert labels_new.shape[0] == labels_old.shape[0]

    for x in prange(rows, nogil=True):
        c_x = labels_new[x]
        for y in range(cols):
            c_y_new = labels_new[y]
            c_y_old = labels_old[y]
            if c_x == c_y_new:
                inner_new[x, c_x] += kmatrix[x, y]
            if c_x == c_y_old:
                mixed[x, c_x] += kmatrix[x, y]
            if labels_old[x] == c_y_old:
                inner_old[x, c_y_old] += kmatrix[x, y]

    return np.sum(inner_new, axis=0), np.sum(mixed, axis=0), np.sum(inner_old, axis=0)



cpdef long[::1] calc_sizes(
        long[::1] labels, 
        const long n_clusters):
    cdef:
        Py_ssize_t size = labels.shape[0] 
        Py_ssize_t i
        long[::1] sizes = np.zeros(n_clusters, dtype=np.int_)
    for i in range(size):
        sizes[labels[i]] += 1
    return sizes

def calc_sums_full(
        double[:, ::1] kmatrix, 
        long[::1] labels, 
        long n_clusters):
    cdef:
        int size = labels.shape[0]
        double[:, ::1] inner_sums = np.zeros((size, n_clusters), dtype=np.double)
        double[:, ::1] outer_sums = np.zeros((size, n_clusters), dtype=np.double)
        Py_ssize_t i,j
        double kernel_ij
        int label_i, label_j
    
    for i in prange(size, nogil=True):
        label_i = labels[i]
        for j in range(size):
            label_j = labels[j]
            kernel_ij = kmatrix[i, j]
            outer_sums[i, label_j] += kernel_ij      
            if label_i == label_j:
                inner_sums[i, label_j] += kernel_ij
    return np.asarray(outer_sums), np.sum(inner_sums, axis=0)







def fill_empty_clusters(long[::1] labels, const long n_clusters, return_sizes=True):
    cdef Py_ssize_t i, index
    cdef long l_size = labels.size
    
    # np.random.seed(0)
    rng = np.random.default_rng(0)

    while True:
        cluster_sizes = np.asarray(calc_sizes(labels, n_clusters), dtype=np.int_)
        empty_cluster_indices = np.flatnonzero(cluster_sizes == 0)
        amount_empty_clusters = empty_cluster_indices.size
        if(amount_empty_clusters == 0):
            break
        for i in range(amount_empty_clusters):
            print("Warning! Empty cluster encountered, consider using different n_cluster. Random element assigned to emtpy cluster")
            # index = np.random.randint(l_size)
            # index = rand() % l_size
            index = rng.integers(l_size)
            labels[index] = empty_cluster_indices[i]
    if return_sizes:
        return labels, np.asarray(cluster_sizes, dtype=np.int_)
    return labels