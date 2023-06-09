'''
Tests to ensure the kernels calculate correctly, via comparing to the results from sklearn.metrics.pairwise kernels.
Tests for input validation etc are done in test_kernel_matrix.
'''
import pytest
import numpy as np
from kernels import rbf_kernel, linear_kernel, sigmoid_kernel, polynomial_kernel
import sklearn.metrics.pairwise as sk

seed = 0
rng = np.random.default_rng(seed)



@pytest.mark.parametrize("xsize", [1, 1000, 1001])
@pytest.mark.parametrize("ysize", [1, 1000, 1001])
@pytest.mark.parametrize("dim", [1, 5, 10])
@pytest.mark.parametrize("variance", [1, 100, 0.1, 0.01])
def test_correctnessrbf_kernel(xsize, ysize, dim, variance):
    X = rng.random((xsize, dim))
    Y = rng.random((ysize, dim))
    gamma = 1/(variance**2)
    assert np.allclose(sk.rbf_kernel(X, Y, gamma), rbf_kernel(X, Y, variance))



@pytest.mark.parametrize("xsize", [1, 1000, 1001])
@pytest.mark.parametrize("ysize", [1, 1000, 1001])
@pytest.mark.parametrize("dim", [1, 5, 10])
def test_correctnesslinear_kernel(xsize, ysize, dim):
    X = rng.random((xsize, dim))
    Y = rng.random((ysize, dim))
    assert np.allclose(sk.linear_kernel(X, Y), linear_kernel(X, Y))


@pytest.mark.parametrize("xsize", [1, 1000, 1001])
@pytest.mark.parametrize("ysize", [1, 1000, 1001])
@pytest.mark.parametrize("dim", [1, 5, 10])
@pytest.mark.parametrize("c", [0, 10, 0.01])
@pytest.mark.parametrize("theta", [0, 1, 20])
def test_correctnesssigmoid_kernel(xsize, ysize, dim, c, theta):
    X = rng.random((xsize, dim))
    Y = rng.random((ysize, dim))
    assert np.allclose(sk.sigmoid_kernel(X, Y, gamma=c, coef0=theta), sigmoid_kernel(X, Y, c, theta))


@pytest.mark.parametrize("xsize", [1, 1000, 1001])
@pytest.mark.parametrize("ysize", [1, 1000, 1001])
@pytest.mark.parametrize("dim", [1, 5, 10])
@pytest.mark.parametrize("c", [0, 10, 0.01])
@pytest.mark.parametrize("d", [0, 1, 20])
def test_correctnesspolynomial_kernel(xsize, ysize, dim, c, d):
    X = rng.random((xsize, dim))
    Y = rng.random((ysize, dim))
    assert np.allclose(sk.polynomial_kernel(X, Y, gamma=1, coef0=c, degree=d), polynomial_kernel(X, Y, c, d))

