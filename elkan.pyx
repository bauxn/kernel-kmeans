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
        long[::1] sizes_old,
        double[::1] inner_sums,
        long n_clusters):
    '''
    updates the lower bounds estimation 

    computes necessary terms (on detailed info of inner_sums etc see 
    thesis) to update lower bounds estimation and then
    uses the algorithm described in KKMeans. (NOT vanilla elkan's
    algorithm!)

    parameters
    ----------
    kernel_matrix: ndarray of shape(n_samples, n_samples), dtype=np.double
    l_bounds: ndarray of shape(n_samples, n_clusters), dtype=np.double
        distance estimations
    center_dists: ndarray of shape(n_samples, n_clusters), dtype=np.double
        cumulative movements of each cluster relative for each datapoint
    labels: ndarray of shape(n_samples), dtype=np.int_
    labels_old: ndarray of shape(n_samples), dtype=np.int_
        labels of prev iteration, used to update center_dists
    sizes: ndarray of shape(n_samples), dtype=np.int_
    inner_sums: ndarray of shape(n_clusters)
    n_clusters: int

    returns
    -------
    l_bounds: ndarray of shape(n_samples, n_clusters)
        updated lower bounds
    inner_sums: ndarray of shape(n_clusters)
    sizes: ndarray of shape(n_clusters)
    center_dists: ndarray of shape(n_clusters, n_clusters)
        updated center_dists
    '''


    inner_sums_old = inner_sums
    inner_sums_mixed, inner_sums = _calc_inner_sums_mixed(kernel_matrix, labels, labels_old, 
                                                          n_clusters, return_inner_new=True)
    center_dists += _calc_center_dists(inner_sums, inner_sums_mixed, inner_sums_old, sizes, sizes_old)
    l_bounds = _est_lower_bounds(kernel_matrix, l_bounds, center_dists, labels, sizes, inner_sums)
    return np.asarray(l_bounds), np.asarray(inner_sums), np.asarray(sizes), np.asarray(center_dists)


def _est_lower_bounds(
        double[:, ::1] kernel_matrix, 
        double[:, ::1] l_bounds, 
        double[:, ::1] center_dists, 
        long[::1] labels, 
        long[::1] sizes, 
        double[::1] inner_sums):
    '''
    heuristic to update lower bounds

    at first, updates the lower bound from 
    each x to its assigned cluster-cetner c_x
    (and sets the relative movement from c_x to x to 0)
    then checks for each center c if it has moved enough
    to be possibly closer to x than c_x. if yes, the lower_bound
    from x to c is updated and the relative movement from c to x
    set to 0.
    '''
    cdef:
        Py_ssize_t i, j, labels_i
        double outer_sum

    assert l_bounds.shape[0] == labels.shape[0]
    assert l_bounds.shape[1] == inner_sums.shape[0]

    for i in prange(l_bounds.shape[0], nogil=True):
        labels_i = labels[i]
        outer_sum = _calc_outer_sum_single(kernel_matrix, i, labels_i, labels)
        # updating lower bounds from x to c_x
        l_bounds[i, labels_i] = (
            kernel_matrix[i, i] 
            - 2 * outer_sum / sizes[labels_i] 
            + inner_sums[labels_i] / sizes[labels_i]**2)
        center_dists[i, labels_i] = 0  # relative movement set to 0
        for j in range(l_bounds.shape[1]):
            if sqrt(l_bounds[i, j]) - center_dists[i, j] <= sqrt(l_bounds[i, labels_i]): 
                outer_sum = _calc_outer_sum_single(kernel_matrix, i, j, labels)
                l_bounds[i, j] = (kernel_matrix[i, i] 
                                 - 2 * outer_sum / sizes[j] 
                                 + inner_sums[j] / sizes[j]**2)
                center_dists[i, j] = 0
    return l_bounds


def _calc_center_dists(
        double[::1] inner_sums_new, 
        double[::1] inner_sums_mixed, 
        double[::1] inner_sums_old, 
        long[::1] sizes_new, 
        long[::1] sizes_old):
    '''
    mathematical helper function

    returns
    ------
    dists: ndarray of shape(n_clusters), dtype=np.double
        distances between new and old cluster centers
        dists[i] = ||center_i_new - center_i_old||
        NOT PAIRWISE!
    '''
    cdef:
        Py_ssize_t i
        Py_ssize_t n_clusters = inner_sums_new.shape[0]
        double[::1] dists = np.zeros(n_clusters)
        long new_size, old_size
        double mixed_sum, new_sum, old_sum

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
        dists[i] = sqrt(new_sum / new_size**2 
                        - 2*mixed_sum / (new_size * old_size) 
                        + old_sum / old_size**2)
    
    return np.asarray(dists) 
    



def _calc_inner_sums_mixed(
        double[:, ::1] kernel_matrix, 
        long[::1] labels, 
        long[::1] labels_old, 
        long n_clusters, 
        return_inner_new=True):
    '''
    Mathematical helper function 

    Notes
    -----
    sums_mixed:
        sums_mixed[c, c'] = sum(K(x,y)) for x in c for y in c' 
        when return it is summed over the columns. then,
        sums_mixed[c] = sum(K(x,y) for x in c_i_new for y in c_i_old))
        (therefore sums_mixed, as it is pairwise over old and new centers)
    '''
    cdef:
        Py_ssize_t size = kernel_matrix.shape[0]
        double[:, ::1] sums_mixed = np.zeros((size, n_clusters))
        double[:, ::1] sums_new = np.zeros((size, n_clusters))
        Py_ssize_t i, j
        long labels_i
    
    assert size == kernel_matrix.shape[1]

    for i in prange(size, nogil=True):
        labels_i = labels[i]
        for j in range(size):
            if labels_i == labels[j]:
                sums_new[i, labels[j]] += kernel_matrix[i, j]
            if labels_i == labels_old[j]:
                sums_mixed[i, labels_old[j]] += kernel_matrix[i, j]
    if return_inner_new:
        return np.sum(sums_mixed, axis=0), np.sum(sums_new, axis=0)
    return np.sum(sums_mixed, axis=0)




def start_elkan(
        sq_distances, 
        double[:, ::1] kernel_matrix,
        long[::1] labels, 
        long n_clusters,
        long[::1] cluster_sizes):
    '''
    Init function for elkan

    functionally equivalent to update_lloyd, included here 
    to make code less dependent.
    Computes the square distances between all samples and all
    centers.

    Parameters
    ----------
    sq_distances: ndarray of shape(n_samples, n_clusters), dtype=np.double
        array on which the distances should be built upon. the one necessary
        here is provided by KKMeans.build_starting_distance()
    kernel_matrix: ndarray of shape(n_samples, n_samples), dtype=np.double
    labels: ndarray of shape(n_samples), dtype=np.int_
    n_clusters: int

    Returns
    -------
    sq_distances: ndarray of shape(n_samples, n_clusters), dtype=np.double
        exact, updated square distances
    inner_sums: ndarray of shape(n_clusters), dtype=np.double
        mathematical term used for prediction functionality
    cluster_sizes: ndarray of shape(n_clusters), dtype=np.int_
    '''
    cdef Py_ssize_t cluster

    
    sq_distances = np.asarray(sq_distances, dtype=np.double)
    outer_sums, inner_sums = _calc_sums_full(kernel_matrix, labels, n_clusters)
    
    for cluster in range(n_clusters):
        size = cluster_sizes[cluster]
        outer_sum = outer_sums[:, cluster]
        inner_sum = inner_sums[cluster] 
        sq_distances[:,  cluster] += (-2 * outer_sum / size 
                                     + inner_sum / size**2)
    return np.asarray(sq_distances), np.asarray(inner_sums), np.asarray(cluster_sizes)


def _calc_sums_full(double[:, ::1] kernel_matrix, long[::1] labels, long n_clusters):
    '''
    Mathematical helper function

    Returns
    -------
    outer_sums: ndarray of shape(n_samples, n_clusters)
        outer_sums[i, j] == sum(K(data[i], data[x])) for x in cluster j
        In words, the sum of the kernel of datapoint X[i] and all datapoints
        in cluster number j.
    inner_sums: ndarray of shape(n_clusters)
        "inner 'distance' sum"; 
        inner_sums[i] = sum(K(x,y)) for all x, y in cluster i
    '''
    cdef:
        Py_ssize_t rows = kernel_matrix.shape[0]
        Py_ssize_t cols = kernel_matrix.shape[1]
        double[:, ::1] inner_sum = np.zeros((rows, n_clusters), dtype=np.double)
        double[:, ::1] outer_sum = np.zeros((rows, n_clusters), dtype=np.double)
        Py_ssize_t i,j
        int label_i, label_j
    for i in prange(rows, nogil=True):
        label_i = labels[i]
        for j in range(cols):
            label_j = labels[j]
            outer_sum[i, label_j] += kernel_matrix[i, j]      
            if label_i == label_j:
                inner_sum[i, label_j] += kernel_matrix[i, j]
    return np.asarray(outer_sum), np.sum(inner_sum, axis=0)


cpdef double _calc_outer_sum_single(
        double[:, ::1] kernel_matrix,
        long elem_index, 
        long cluster_index, 
        long[::1] labels) nogil:
    '''
    mathematical helper function
    
    given sample x and cluster c:
    computes sum(K(x, y)) for all y in c 
    '''
    cdef:
        double outer_sum = 0.
        Py_ssize_t i
        Py_ssize_t size = len(labels)
    for i in range(size):
        if labels[i] == cluster_index:
            outer_sum += kernel_matrix[elem_index, i]
    return outer_sum
