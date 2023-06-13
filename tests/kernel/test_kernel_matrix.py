import pytest
import numpy as np
from kernels import build_kernel_matrix as km
from sklearn.metrics.pairwise import pairwise_kernels as pk

seed = 0
rng = np.random.default_rng(seed)

################# TEST DEFAULT PARAMS ###############
@pytest.mark.parametrize("x_shape", [(20,10), (10,20), (10,10)])
@pytest.mark.parametrize("variance", [0.5])
@pytest.mark.parametrize("c", [2])
@pytest.mark.parametrize("d", [4])
@pytest.mark.parametrize("theta", [1])
def test_Y_is_None(x_shape, variance, c, d, theta):
    X = rng.random(size=x_shape)
    assert(np.allclose(km(X), pk(X)))
    assert(np.allclose(km(X, kernel="rbf"), pk(X, metric="rbf")))
    assert(np.allclose(km(X, kernel="sigmoid"), pk(X, metric="sigmoid", gamma=1)))
    assert(np.allclose(km(X, kernel="polynomial"), pk(X, metric="polynomial", gamma=1)))

    assert(np.allclose(km(X, kernel="rbf", variance=variance), pk(X, metric="rbf", gamma=1/variance**2)))
    assert(np.allclose(km(X, kernel="sigmoid", c=c, theta=theta), pk(X, metric="sigmoid", gamma=c, coef0=theta)))
    assert(np.allclose(km(X, kernel="polynomial", c=c, d=d), pk(X, metric="polynomial", gamma=1, degree=d, coef0=c)))

################# TEST CONVERSION ##################
def test_convert_float_1d():
    X = rng.random(size=10)
    Y = rng.random(size=10)
    assert np.allclose(km(X), pk([X])) 
    assert np.allclose(km(X), km([X]))
    assert np.allclose(km(X, Y), pk([X], [Y]))
    assert np.allclose(km(X, Y), km([X], [Y]))
    var = 0.5
    assert np.allclose(km(X, kernel="rbf", variance = var), pk([X], metric="rbf", gamma = 1/(var**2)))
    assert np.allclose(km(X, kernel="rbf", variance = var), km([X], kernel="rbf", variance=0.5))
    assert np.allclose(km(X, Y, kernel="rbf", variance = var), pk([X], [Y], metric="rbf", gamma = 1/(var**2)))
    assert np.allclose(km(X, Y, kernel="rbf", variance = var), km([X], [Y], kernel="rbf", variance=0.5))

def test_convert_integers_1d():
    X = rng.integers(low=0, high=2000, size=10)
    Y = rng.integers(low=0, high=2000, size=10)
    assert np.allclose(km(X), pk([X])) 
    assert np.allclose(km(X), km([X]))
    assert np.allclose(km(X, Y), pk([X], [Y]))
    assert np.allclose(km(X, Y), km([X], [Y]))
    var = 0.5
    assert np.allclose(km(X, kernel="rbf", variance = var), pk([X], metric="rbf", gamma = 1/(var**2)))
    assert np.allclose(km(X, kernel="rbf", variance = var), km([X], kernel="rbf", variance=0.5))
    assert np.allclose(km(X, Y, kernel="rbf", variance = var), pk([X], [Y], metric="rbf", gamma = 1/(var**2)))
    assert np.allclose(km(X, Y, kernel="rbf", variance = var), km([X], [Y], kernel="rbf", variance=0.5))

def test_convert_integers_2d():
    X = rng.integers(low=0, high=2000, size=(10,20))
    Y = rng.integers(low=0, high=2000, size=(10, 20))
    assert np.allclose(km(X), pk(X)) 
    assert np.allclose(km(X, Y), pk(X, Y))
    assert np.allclose(km(X, kernel="rbf", variance = 0.5), pk(X, metric="rbf", gamma = 1/0.25))
    assert np.allclose(km(X, Y, kernel="rbf", variance = 0.5), pk(X, Y, metric="rbf", gamma = 1/0.25))


############## TEST REJECTION OF INVALID INPUT ##############

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("scalar", [0, 0., 1, 1., -1, -1.])
def test_reject_single_scalar(scalar):
    km(scalar)


@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("X", ["asdasd", "", ["asd", "cde", "asdads"], ["asd"], [[0., 1., 2., "asd"]], [0., 1., "asd"]])
def test_reject_strings(X):
    km(X)

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("dims", [(1,1,1), (1,1,0), (1,1,1,1)])
def test_reject_too_high_dimension(dims):
    X = rng.random(size=dims)
    km(X)

@pytest.mark.xfail(strict=True, raises=ValueError)
@pytest.mark.parametrize("dims", [(0,), (0), (0,1), (0,1,1), (1,0,1), (1,0,0), (1,1,0)])
def test_reject_zero_dimension(dims):
    X = rng.random(size=dims)
    km(X)

@pytest.mark.xfail(strict=True)
def test_reject_empty_data_zero():
    km([])

@pytest.mark.xfail(strict=True, raises=NotImplementedError)
@pytest.mark.parametrize("kernel", ["", "asdasd", 2, None])
def test_check_invalid_kernel(kernel):
    X = rng.random(size=(5, 5))
    km(X, kernel=kernel)

@pytest.mark.xfail(strict=True)
def test_reject_zero_variance():
    X = rng.random(size=(5,5))
    km(X, kernel="rbf", variance=0)

############ OTHER #############


def test_sigmoid_zero_to_zeroth_power():
    X = (-1,1)
    Y = (1, -1)
    km(X,Y, kernel="sigmoid", c=1, theta=0)


@pytest.mark.parametrize("x_shape", [(20,10), (10,20), (10,10)])
@pytest.mark.parametrize("variance", [0.5])
@pytest.mark.parametrize("c", [2])
@pytest.mark.parametrize("d", [4])
@pytest.mark.parametrize("theta", [1])
def test_X_is_nullmatrix(x_shape, variance, c, d, theta):
    X = np.zeros(shape=x_shape)
    assert(np.allclose(km(X), pk(X)))
    assert(np.allclose(km(X, kernel="rbf"), pk(X, metric="rbf")))
    assert(np.allclose(km(X, kernel="sigmoid"), pk(X, metric="sigmoid", gamma=1)))
    assert(np.allclose(km(X, kernel="polynomial"), pk(X, metric="polynomial", gamma=1)))

    assert(np.allclose(km(X, kernel="rbf", variance=variance), pk(X, metric="rbf", gamma=1/variance**2)))
    assert(np.allclose(km(X, kernel="sigmoid", c=c, theta=theta), pk(X, metric="sigmoid", gamma=c, coef0=theta)))
    assert(np.allclose(km(X, kernel="polynomial", c=c, d=d), pk(X, metric="polynomial", gamma=1, degree=d, coef0=c)))

