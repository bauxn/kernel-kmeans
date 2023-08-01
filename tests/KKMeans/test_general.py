from KKMeans import KKMeans
from sklearn.datasets import make_blobs
import numpy as np
import pytest


def test_zero_ncluster_fail():
    try:
        KKMeans(0)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.parametrize("n_clusters", [1, 10, 100])
def test_wrong_amount_centers(n_clusters):
    n_samples = 100
    n_features = 3
    bad_nclusters = 2
    _, __, centers = make_blobs(
        n_samples, n_features, centers=bad_nclusters, return_centers=True
    )
    try:
        kkm = KKMeans(n_clusters, init=centers)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.parametrize("n_clusters", [0, 10, 100])
def test_given_centers_empty_fail(n_clusters):
    centers = np.asarray([])
    try:
        kkm = KKMeans(n_clusters, init=centers)
    except ValueError:
        pass
    else:
        assert False


@pytest.mark.parametrize("n_samples", [2000])
@pytest.mark.parametrize("n_features", [1, 2, 100])
@pytest.mark.parametrize("n_clusters", [1, 2, 50])
def test_given_centers(n_samples, n_features, n_clusters):
    data, _ = make_blobs(n_samples, n_features, centers=n_clusters)
    kkm = KKMeans(n_clusters)
    kkm.fit(data)


@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10]])
def test_bad_n_clusters(param):
    try:
        KKMeans(n_clusters=param)
    except ValueError as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize(
    "param", [None, "asd", -1, 0, 0.1, 10.5, [], (), None, ([]), ["kernel"]]
)
def test_bad_init(param):
    try:
        KKMeans(init=param)
    except (ValueError, NotImplementedError) as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize("n_clusters", [1, 10, 200])
def test_string_init_fail(n_clusters):
    init = ["string"] * n_clusters
    try:
        KKMeans(n_clusters, init=init)
    except ValueError as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10], ["kernel"]])
def test_n_init(param):
    try:
        KKMeans(n_init=param)
    except (ValueError, NotImplementedError) as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize(
    "param", [[], (), None, ([]), "asd", -1, 0, [10], ["kernel"], 0.1, 10.5]
)
def test_max_iter(param):
    try:
        KKMeans(max_iter=param)
    except ValueError as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize(
    "param",
    [
        [],
        (),
        None,
        ([]),
        "asd",
        [10],
        ["kernel"],
        pytest.param(-1, marks=pytest.mark.xfail(reason="is set to 0")),
    ],
)
def test_tol(param):
    try:
        KKMeans(tol=param)
    except ValueError as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize("param", [[], (), None, ([]), "asd", -1, 0, [10], ["kernel"]])
def test_q_metric(param):
    try:
        KKMeans(q_metric=param)
    except (ValueError, TypeError, NotImplementedError) as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize("param", [None, [], [()], "asd", 10.5, -1, ("asd"), [1, 10.5]])
def test_algorithm(param):
    try:
        KKMeans(algorithm=param)
    except (ValueError, NotImplementedError) as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False


@pytest.mark.parametrize(
    "param", [None, [], [()], "asd", 10.5, -1, ("asd"), [1, 10.5], ["kernel"]]
)
def test_kernel(param):
    try:
        KKMeans(kernel=param)
    except ValueError as err:
        print(err)
        print(err.args)
        pass
    else:
        assert False
