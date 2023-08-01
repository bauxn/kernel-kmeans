import numpy as np
from cython.parallel import prange
from libc.math cimport exp, tanh, fabs

def build_kernel_matrix(X, Y=None, kernel="linear", gamma=1, c_0=0, d=3, variance=1):
    '''
    wrapper function to build kernel matrix

    Parameters
    ----------
    X: ndarray of shape(n_samples, n_features), dtype=np.double
        data to build kernel matrix from
    Y: ndarray of shape(*, n_features), dtype=np.double, optional
        If None, Y=X. data to compute kernel matrix 
        against, eg kernel_matrix[i,j] = K(X[i], Y[j])
        Y=X leads to expected behaviour
    kernel: {"linear", "rbf", "gaussian", "polynomial", "sigmoid", "laplacian"}
            default="linear"
    gamma: double, default=1
        param for kernel
    c_0: double, default=0
        param for kernel
    d: double, default=3
        param for kernel
    variance: double, default=1
        param for gaussian kernel.
        is passed to rbf with gamma=1/variance
    
    Returns
    -------
    kernel_matrix: ndarray, dtype=np.double 
        of shape(len(X), len(X)) if Y=None
        of shape(len(X), len(Y)) else
    '''
    if Y is None:
        Y = X
    X, Y = _sanitize_data(X, Y)
    if kernel == "linear":
        return np.asarray(linear_kernel(X, Y))
    elif kernel == "rbf":
        return np.asarray(rbf_kernel(X, Y, gamma))
    elif kernel == "gaussian":
        if np.isclose(variance, 0, atol=1e-8):
            raise ValueError("Variance too close to 0 for gaussian kernel!")
        return np.asarray(rbf_kernel(X, Y, 1/variance))
    elif kernel == "polynomial":
        return np.asarray(polynomial_kernel(X, Y, d, gamma, c_0))
    elif kernel == "sigmoid":
        return np.asarray(sigmoid_kernel(X, Y, gamma, c_0))
    elif kernel == "laplacian":
        return np.asarray(laplacian_kernel(X, Y, gamma))
    else:
        raise NotImplementedError(str(kernel) + " kernel not implemented")


def _sanitize_data(X, Y):
    X, Y = _cast_to_ndarray(X, Y)
    X, Y = _check_dimensionality(X, Y)
    return X, Y

def _check_dimensionality(X, Y):
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
    X, Y = np.array(X, dtype=np.double), np.array(Y, dtype=np.double)
    return X, Y


cpdef linear_kernel(double[:, ::1] X, double[:, ::1] Y):
    '''K(x,y) == x.T * y'''
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


cpdef rbf_kernel(double[:, ::1] X, double[:, ::1] Y, double gamma):
    '''K(x,y) == exp(-gamma * ||x - y||^2)'''
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double sq_euclidian
        double[:, ::1] matrix = np.zeros((x_size, y_size))
    
    if np.isclose(gamma, 0, atol=1e-6):
        raise ValueError("Gamma too close to 0 for rbf kernel")

    assert dim == Y.shape[1]

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            sq_euclidian = 0
            for k in range(dim):
                sq_euclidian = sq_euclidian + (X[i,k] - Y[j,k]) ** 2
            matrix[i,j] = exp(-gamma * sq_euclidian)
    return matrix  

cpdef sigmoid_kernel(double[:, ::1] X, double[:, ::1] Y, double gamma, double c_0):
    '''k(x,y) == tanh(gamma * (x.T * y) + c_0)'''
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    assert dim==Y.shape[1]

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod + X[i,k] * Y[j,k]
            matrix[i,j] = tanh(gamma * dot_prod + c_0)
    return matrix

cpdef polynomial_kernel(double[:, ::1] X, double[:, ::1] Y, long d, double gamma, double c_0):
    '''K(x,y) == (gamma * (x.T * y) + c_0)**d'''
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double dot_prod
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    assert dim == Y.shape[1]

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            dot_prod = 0
            for k in range(dim):
                dot_prod = dot_prod + X[i,k] * Y[j,k]
            matrix[i,j] = (gamma * dot_prod + c_0) ** d
    return matrix


cpdef laplacian_kernel(double[:, ::1] X, double[:, ::1] Y, double gamma):
    '''K(x,y) == exp(-gamma * ||x - y||_1)'''
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double one_norm
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    assert dim == Y.shape[1]

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            one_norm = 0
            for k in range(dim):
                one_norm = one_norm + fabs(X[i,k] - Y[j,k])
            matrix[i,j] = exp(-gamma * one_norm)
    return matrix

cpdef chi_squared_kernel(double[:, ::1] X, double [:, ::1] Y, double gamma):
    '''K(x, y) == exp(-gamma * sum_i((x[i] - y[i])^2)/(x[i] + y[i]))'''
    cdef:
        Py_ssize_t x_size = X.shape[0]
        Py_ssize_t y_size = Y.shape[0]
        Py_ssize_t dim = X.shape[1]
        Py_ssize_t i,j,k
        double summand
        double[:, ::1] matrix = np.zeros((x_size, y_size))

    assert dim == Y.shape[1]

    for i in prange(x_size, nogil=True):
        for j in range(y_size):
            summand = 0
            for k in range(dim):
                summand = summand + (X[i, k] - Y[j, k])**2 / (X[i, k] + Y[j, k])
            matrix[i, j] = summand

    return matrix