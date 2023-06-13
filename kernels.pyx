# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange
from libc.math cimport exp, tanh

def build_kernel_matrix(X, Y=None, kernel="linear", variance=None, c=1, d=3, theta=1):

    X, Y = _sanitize_data(X, Y)
    if variance is None:
        variance = np.sqrt(X.shape[1])
    if kernel == "linear":
        return np.asarray(linear_kernel(X, Y))
    elif kernel == "rbf":
        return np.asarray(rbf_kernel(X, Y, variance))
    elif kernel == "polynomial":
        return np.asarray(polynomial_kernel(X, Y, c, d))
    elif kernel == "sigmoid":
        return np.asarray(sigmoid_kernel(X, Y, c, theta))
    else:
        raise NotImplementedError(str(kernel) + " kernel not implemented")

def _sanitize_data(X, Y):
    if Y is None:
        Y = X
    X, Y = _cast_to_ndarray(X, Y)
    X, Y = _check_dimensions(X, Y)
    return X, Y

def _check_dimensions(X, Y):
    if len(X.shape) == 0 or len(Y.shape) == 0:
        raise ValueError("X and Y need to be 1-d or 2-d (arraylikes)")
    if 0 in X.shape or 0 in Y.shape:
        raise ValueError("Data needs to have at least one sample and one dimension")
    if len(X.shape) == 1:
        X = np.array([X])
    if len(Y.shape) == 1:
        Y = np.array([Y])
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension mismatch")
    if len(X.shape) > 2 or len(Y.shape) > 2:
        raise ValueError("X and Y need to be 1-d or 2-d")
    return X, Y

def _cast_to_ndarray(X, Y):
    X, Y = np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)
    return X, Y


cpdef linear_kernel(double[:, ::1] X, double[:, ::1] Y):
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod +  X[i,k] * Y[j, k]
            matrix[i,j]= dot_prod
    return matrix

    
cpdef rbf_kernel(double[:, ::1] X, double[:, ::1] Y, double variance):
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double sq_euclidian
        double[:, ::1] matrix = np.zeros((x_size, y_size))
        double gamma
    
    if variance == 0.:
        raise ValueError("variance must not be 0")
    gamma = 1/(variance ** 2)

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            sq_euclidian = 0
            for k in range(dim):
                sq_euclidian = sq_euclidian + (X[i,k] - Y[j,k]) ** 2
            matrix[i,j] = exp(-gamma * sq_euclidian)
    return matrix

cpdef sigmoid_kernel(double[:, ::1] X, double[:, ::1] Y, double c, double theta):
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod + X[i,k] * Y[j,k]
            matrix[i,j] = tanh(c * dot_prod + theta)
    return matrix

cpdef polynomial_kernel(double[:, ::1] X, double[:, ::1] Y, double c, int d):
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod + X[i,k] * Y[j,k]
            matrix[i,j] = (dot_prod + c) ** d
    return matrix

