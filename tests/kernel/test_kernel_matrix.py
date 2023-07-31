import pytest
import numpy as np
from KKMeans.kernels import build_kernel_matrix as km
from sklearn.metrics.pairwise import pairwise_kernels as ctrl
from tests.pytest_utils import RNG


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("x_size", [1, 101])
@pytest.mark.parametrize("y_size", [1, 101])
@pytest.mark.parametrize("dim", [1, 20])
@pytest.mark.parametrize("c_0", [0])
@pytest.mark.parametrize("d", [3])
@pytest.mark.parametrize("gamma", [1])
def test_def_params(max_data, x_size, y_size, dim, c_0, d, gamma):
    X = RNG.random((x_size, dim)) * max_data
    Y = RNG.random((y_size, dim)) * max_data
    assert np.allclose(
        km(X, Y, "linear"),
        ctrl(X, Y, "linear"))
    assert np.allclose(
        km(X, Y, "rbf"),
        ctrl(X, Y, "rbf", gamma=gamma))
    assert np.allclose(
        km(X, Y, "polynomial"),
        ctrl(X, Y, "poly", gamma=gamma, coef0=c_0, degree=d))
    assert np.allclose(
        km(X, Y, "sigmoid"),
        ctrl(X, Y, "sigmoid", gamma=gamma, coef0=c_0))
    assert np.allclose(
        km(X, Y, "laplacian"),
        ctrl(X, Y, "laplacian", gamma=gamma))

@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("x_size", [1, 101])
@pytest.mark.parametrize("dim", [1, 20])
@pytest.mark.parametrize("c_0", [7])
@pytest.mark.parametrize("d", [5])
@pytest.mark.parametrize("gamma", [0.1])
def test_Y_is_None(max_data, x_size, dim, c_0, d, gamma):
    X = RNG.random((x_size, dim)) * max_data
    Y = None
    assert np.allclose(
        km(X, Y, "linear"),
        ctrl(X, Y, "linear"))
    assert np.allclose(
        km(X, Y, "rbf", gamma=gamma),
        ctrl(X, Y, "rbf", gamma=gamma))
    assert np.allclose(
        km(X, Y, "polynomial", gamma=gamma, c_0=c_0, d=d),
        ctrl(X, Y, "polynomial", gamma=gamma, coef0=c_0, degree=d))
    assert np.allclose(
        km(X, Y, "sigmoid", gamma=gamma, c_0=c_0),
        ctrl(X, Y, "sigmoid", gamma=gamma, coef0=c_0))
    assert np.allclose(
        km(X, Y, "laplacian", gamma=gamma),
        ctrl(X, Y, "laplacian", gamma=gamma))

@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("gamma", [0.5])
def test_1dfloat_to_2dfloat(n_samples, gamma):
    '''
    tests if 1d input is converted properly
    
    the utility of calculating the kernel of a single
    input vector is not implemented in sklearn
    '''
    X = RNG.random(n_samples)
    Y = RNG.random(n_samples)
    assert np.allclose(km(X), ctrl([X])) 
    assert np.allclose(km(X), km([X]))
    assert np.allclose(km(X, Y), ctrl([X], [Y]))
    assert np.allclose(km(X, Y), km([X], [Y]))
    assert np.allclose(
        km(X, kernel="rbf", gamma=gamma), 
        ctrl([X], metric="rbf", gamma=gamma))
    assert np.allclose(
        km(X, Y, kernel="rbf", gamma=gamma), 
        ctrl([X], [Y], metric="rbf", gamma=gamma))


@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("max_data", [1000])
@pytest.mark.parametrize("dim", [10])
@pytest.mark.parametrize("gamma", [0.5])
def test_int_to_float(n_samples, max_data, dim, gamma):
    '''tests if cython converts int properly'''
    X = RNG.integers(max_data, size=(n_samples, dim))
    Y = RNG.integers(max_data, size=(n_samples, dim))
    assert np.allclose(km(X), ctrl(X)) 
    assert np.allclose(km(X), km(X))
    assert np.allclose(km(X, Y), ctrl(X, Y))
    assert np.allclose(km(X, Y), km(X, Y))
    assert np.allclose(
        km(X, kernel="rbf", gamma=gamma), 
        ctrl(X, metric="rbf", gamma=gamma))
    assert np.allclose(
        km(X, Y, kernel="rbf", gamma=gamma), 
        ctrl(X, Y, metric="rbf", gamma=gamma))


############## TEST REJECTION OF INVALID INPUT ##############

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("scalar", [0, 0., 1, 1., -1, -1.])
def test_reject_single_scalar(scalar):
    km(scalar)


@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("X", [["asdasd"], "", ["asd", "cde", "asdads"], ["asd"], [[0., 1., 2., "asd"]], [0., 1., "asd"]])
def test_reject_strings(X):
    km(X)

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("dims", [(1,1,1), (1,1,0), (1,1,1,1), (0,0,0)])
def test_reject_too_high_dimension(dims):
    X = RNG.random(size=dims)
    km(X)

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("dims", [(0,), (0), (0,1), (0,1,1), (1,0,1), (1,0,0), (1,1,0), (0, 0)])
def test_reject_zero_dimension(dims):
    X = RNG.random(size=dims)
    km(X)

@pytest.mark.xfail(strict=True)
def test_reject_empty_data_zero():
    km([])

@pytest.mark.xfail(strict=True, raises=NotImplementedError)
@pytest.mark.parametrize("kernel", ["", "asdasd", 2, None])
def test_check_invalid_kernel(kernel):
    X = RNG.random(size=(5, 5))
    km(X, kernel=kernel)

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("variance", [0., 1e-10, 0])
def test_reject_zero_variance(variance):
    X = RNG.random(size=(5,5))
    km(X, kernel="gaussian", variance=variance)



def test_polynomial_zero_to_zeroth_power():
    X = [(0, 0)]
    Y = [(0, 0)]
    assert np.allclose(
        km(X, Y, kernel="polynomial", c_0=0, gamma=0, d=0),
        ctrl(X, Y, metric="poly", coef0=0, gamma=0, degree=0))


@pytest.mark.parametrize("n_samples", [1, 1000])
@pytest.mark.parametrize("dim", [1,100])
@pytest.mark.parametrize("gamma", [1])
def test_X_is_nullmatrix(n_samples, dim, gamma):
    X = np.zeros((n_samples, dim))
    assert np.allclose(km(X), ctrl(X))
    assert np.allclose(
        km(X, kernel="rbf", gamma=gamma), 
        ctrl(X, metric="rbf", gamma=gamma))
