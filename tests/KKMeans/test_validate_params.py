from KKMeans import KKMeans
import pytest

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10]])
def test_n_clusters(param):
    KKMeans(n_clusters=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", [None, "asd", -1, 0, 0.1, 10.5])
def test_init(param):
    KKMeans(init=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10], ["kernel"]])
def test_n_init(param):
    KKMeans(n_init=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10], ["kernel"],0.1, 10.5])
def test_max_iter(param):
    KKMeans(max_iter=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", [10], ["kernel"]])
def test_tol(param):
    KKMeans(tol=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10], ["kernel"]])
def test_q_metric(param):
    KKMeans(q_metric=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", ["asd", 10.5, -1, ("asd"), [1, 10.5]])
def test_rng(param):
    KKMeans(rng=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", ["asd", 10.5, -1, ("asd"), [1, 10.5]])
def test_algorithm(param):
    KKMeans(algorithm=param)

@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("param", ["asd", 10.5, -1, ("asd"), [1, 10.5], ["kernel"]])
def test_kernel(param):
    KKMeans(algorithm=param)