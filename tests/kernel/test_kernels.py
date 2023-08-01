import pytest
import numpy as np
from KKMeans.kernels import (
    rbf_kernel,
    linear_kernel,
    sigmoid_kernel,
    polynomial_kernel,
    laplacian_kernel,
)
import sklearn.metrics.pairwise as ctrl
from tests.pytest_utils import RNG

"""
tests only if kernels return correct result, no input validation tested here
gaussian kernel is special case of rbf kernel therefore not tested here
"""


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("xsize", [1, 2000])
@pytest.mark.parametrize("ysize", [1, 2000])
@pytest.mark.parametrize("dim", [1, 10])
def test_linear_kernel(max_data, xsize, ysize, dim):
    X = RNG.random((xsize, dim)) * max_data
    Y = RNG.random((ysize, dim)) * max_data
    assert np.allclose(ctrl.linear_kernel(X, Y), linear_kernel(X, Y))


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("xsize", [1, 2000])
@pytest.mark.parametrize("ysize", [1, 2000])
@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("gamma", [1, 100, 0.01])
def test_rbf_kernel(max_data, xsize, ysize, dim, gamma):
    X = RNG.random((xsize, dim)) * max_data
    Y = RNG.random((ysize, dim)) * max_data
    assert np.allclose(ctrl.rbf_kernel(X, Y, gamma), rbf_kernel(X, Y, gamma))


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("xsize", [1, 2001])
@pytest.mark.parametrize("ysize", [1, 2001])
@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("c_0", [0, 10, 0.01])
@pytest.mark.parametrize("gamma", [0, 1, 20])
def test_sigmoid_kernel(max_data, xsize, ysize, dim, c_0, gamma):
    X = RNG.random((xsize, dim)) * max_data
    Y = RNG.random((ysize, dim)) * max_data
    assert np.allclose(
        ctrl.sigmoid_kernel(X, Y, gamma, c_0), sigmoid_kernel(X, Y, gamma, c_0)
    )


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("xsize", [1, 2001])
@pytest.mark.parametrize("ysize", [1, 2001])
@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("c_0", [0, 0.01])
@pytest.mark.parametrize("d", [0, 20])
@pytest.mark.parametrize("gamma", [0, 1, 5])
def test_polynomial_kernel(max_data, xsize, ysize, dim, c_0, d, gamma):
    X = RNG.random((xsize, dim)) * max_data
    Y = RNG.random((ysize, dim)) * max_data
    assert np.allclose(
        ctrl.polynomial_kernel(X, Y, d, gamma, c_0),
        polynomial_kernel(X, Y, d, gamma, c_0),
    )


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("xsize", [1, 2001])
@pytest.mark.parametrize("ysize", [1, 2001])
@pytest.mark.parametrize("dim", [1, 10])
@pytest.mark.parametrize("gamma", [0, 1, 5])
def test_laplacian_kernel(max_data, xsize, ysize, dim, gamma):
    X = RNG.random((xsize, dim)) * max_data
    Y = RNG.random((ysize, dim)) * max_data
    assert np.allclose(
        ctrl.laplacian_kernel(X, Y, gamma), laplacian_kernel(X, Y, gamma)
    )
