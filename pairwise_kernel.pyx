# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: extra_compile_args = /openmp
import numpy as np
from cython.parallel import prange
from libc.math cimport exp, tanh


def kernel_matrix(X, Y=None, kernel="linear", sigma=1., c=0, d=1, theta=0):
    X, Y = cast_to_ndarray(X, Y)
    if Y is None:
        return kernel_matrix(X, X, kernel)
    res_matrix = None #Used to throw error if still None at end of function
    if kernel == "linear":
        res_matrix = linear_kernel(X, Y)
    elif kernel == "rbf":
        res_matrix = rbf_kernel(X, Y, sigma)
    elif kernel == "polynomial":
        res_matrix = polynomial_kernel(X, Y, c, d)
    elif kernel == "sigmoid":
        res_matrix = sigmoid_kernel(X, Y, c, theta)
    if res_matrix is None:
        raise ValueError(str(kernel) + " kernel not implemented")
    return np.asarray(res_matrix)

def cast_to_ndarray(X, Y=None):
    X = np.array(X)
    if Y is not None:
        Y = np.array(Y)
    return X, Y

def linear_kernel(double[:, ::1] X, double[:, ::1] Y):
    cdef:
        int x_size = X.shape[0]
        int y_size = Y.shape[0]
        int dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matr = np.zeros((x_size, y_size))
    if dim != Y.shape[1]:
        raise ValueError("Dimension mismatch")
    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod +  X[i,k] * Y[j, k]
            matr[i,j]= dot_prod
    return matr

    
def rbf_kernel(double[:, ::1] X, double[:, ::1] Y, double sigma):
    cdef:
        int x_size = X.shape[0]
        int y_size = Y.shape[0]
        int dim = X.shape[1]
        Py_ssize_t i,j,k
        double sq_euclidian
        double[:, ::1] matr = np.zeros((x_size, y_size))
        double gamma = 1/(sigma ** 2)
    if dim != Y.shape[1]:
        raise ValueError("Dimension mismatch")
    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            sq_euclidian = 0
            for k in range(dim):
                sq_euclidian = sq_euclidian + (X[i,k] - Y[j,k]) ** 2
            matr[i,j] = exp(-gamma * sq_euclidian)
    return matr

def sigmoid_kernel(double[:, ::1] X, double[:, ::1] Y, double c, double theta):
    cdef:
        int x_size = X.shape[0]
        int y_size = Y.shape[0]
        int dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matr = np.zeros((x_size, y_size))
    if dim != Y.shape[1]:
        raise ValueError("Dimension mismatch")
    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod + X[i,k] * Y[j,k]
            matr[i,j] = tanh(c * dot_prod + theta)
    return matr

def polynomial_kernel(double[:, ::1] X, double[:, ::1] Y, double c, int d):
    cdef:
        int x_size = X.shape[0]
        int y_size = Y.shape[0]
        int dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matr = np.zeros((x_size, y_size))
    if dim != Y.shape[1]:
        raise ValueError("Dimension mismatch")
    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod + X[i,k] * Y[j,k]
            matr[i,j] = (dot_prod + c) ** d
    return matr
