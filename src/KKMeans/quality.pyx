import numpy as np
from cython.parallel import prange

cdef extern from "float.h":
    cdef double DBL_MAX

def avg_silhouette(double[:, ::1] distances, long[::1] labels):
    '''returns the average silhouette of a dataset'''
    silhouettes = calc_silhouettes(distances, labels)
    return np.sum(silhouettes)/len(silhouettes)

def calc_silhouettes(double[:, ::1] distances, long[::1] labels):
    '''
    calculates the silhouette of each datapoint

    For more details see KKMeans class docstring

    Parameters
    ----------
    distances: ndarray of shape(n_samples, n_clusters), dtype=np.double
        distances (or any measurement on a ratio scale) from each sample
        to each center
    labels: ndarray of shape(n_samples)
    
    Returns
    -------
    silhouettes: ndarray of shape(n_samples)
        the silhouette of each sample
    '''
    cdef:
        Py_ssize_t n_samples = distances.shape[0]
        Py_ssize_t n_clusters = distances.shape[1]
        double[::1] silhouettes = np.zeros(n_samples)
        Py_ssize_t i, j
        long labels_i
        double a, b, max_a_b

    if n_clusters <= 1:
        raise ValueError("Must have at least two clusters for silhouette")
    for i in prange(n_samples, nogil=True):
        labels_i = labels[i]
        a = distances[i, labels_i]
        b = DBL_MAX
        for j in range(n_clusters):
            if j == labels_i:
                continue
            b = min(distances[i, j], b)
        max_a_b = max(a,b)
        if max_a_b == 0:
            # if max_a_b = 0, a == b == 0 as dists always positive 
            silhouettes[i] = 0
            continue
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



