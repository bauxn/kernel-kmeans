import numpy as np
from cython.parallel import prange

cdef extern from "float.h":
    cdef double DBL_MAX


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
        Py_ssize_t data_size = distances.shape[0]
        Py_ssize_t n_clusters = distances.shape[1]
        double[::1] silhouettes = np.zeros(data_size)
        Py_ssize_t i, j
        long labels_i
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



