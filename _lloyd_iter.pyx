# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def calc_sums(const double[:, ::1] kernel_matrix, const long long[::1] labels, const int n_clusters):
    cdef:
        int DATA_LEN = kernel_matrix.shape[0]
        double[:, ::1] inner_cluster_terms = np.zeros((DATA_LEN, n_clusters), dtype=np.float64)
        double[:, ::1] element_cluster_terms = np.zeros((DATA_LEN, n_clusters), dtype=np.float64)
        int[:, ::1] n_cluster_elements = np.zeros((DATA_LEN, n_clusters), dtype=np.int32)
        int i,j
        double kernel_ij
        int label_i, label_j
    for i in prange(DATA_LEN, nogil=True):
        label_i = labels[i]
        n_cluster_elements[i, label_i] += 1
        for j in range(DATA_LEN):
            label_j = labels[j]
            kernel_ij = kernel_matrix[i, j]
            element_cluster_terms[i, label_j] += kernel_ij      
            if label_i == label_j:
                inner_cluster_terms[i, label_j] += kernel_ij
    

    return np.asarray(element_cluster_terms), np.sum(inner_cluster_terms, axis=0), np.sum(n_cluster_elements, axis = 0)