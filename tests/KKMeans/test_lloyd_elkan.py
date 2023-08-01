import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from KKMeans import KKMeans


@pytest.mark.parametrize("n_predicts", [1, 100])
@pytest.mark.parametrize("n_samples", [1, 10, 1000])
@pytest.mark.parametrize("n_init", [2])
@pytest.mark.parametrize("n_clusters", [1, 200])
@pytest.mark.parametrize("tol", [0, 1e-4])
@pytest.mark.parametrize("n_features", [1, 5])
@pytest.mark.parametrize("seed", list(range(3)))
@pytest.mark.parametrize("gamma", [0.1])
@pytest.mark.parametrize("c_0", [1])
@pytest.mark.parametrize("d", [3])
@pytest.mark.parametrize(
    "kernel", ["linear", "rbf", "polynomial", "gaussian", "laplacian"]
)
def test_lloyd_elkan(
    n_samples,
    n_init,
    n_clusters,
    tol,
    n_features,
    seed,
    gamma,
    c_0,
    d,
    kernel,
    n_predicts,
):
    if n_clusters > n_samples:
        pytest.xfail()
    data, labels = make_blobs(
        n_samples, n_features, centers=n_clusters, random_state=seed
    )
    lloyd = KKMeans(
        n_clusters,
        n_init=n_init,
        tol=tol,
        rng=seed,
        algorithm="lloyd",
        gamma=gamma,
        d=d,
        c_0=c_0,
        kernel=kernel,
    )
    elkan = KKMeans(
        n_clusters,
        n_init=n_init,
        tol=tol,
        rng=seed,
        algorithm="elkan",
        gamma=gamma,
        d=d,
        c_0=c_0,
        kernel=kernel,
    )
    lloyd.fit(data)
    elkan.fit(data)
    assert all(lloyd.labels_ == elkan.labels_)
    assert np.isclose(lloyd.quality_, elkan.quality_)

    assert all(lloyd.predict(data) == lloyd.labels_)
    assert all(elkan.predict(data) == elkan.labels_)

    pred, _ = make_blobs(n_predicts, n_features, random_state=seed)
    l_pred = lloyd.predict(pred)
    e_pred = elkan.predict(pred)
    assert all(l_pred == e_pred)


@pytest.mark.parametrize("n_samples", [100])
@pytest.mark.parametrize("n_clusters", [101, 200, 1000])
@pytest.mark.parametrize("n_features", [1, 5])
@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("gamma", [0.1])
@pytest.mark.parametrize("c_0", [1])
@pytest.mark.parametrize("d", [3])
@pytest.mark.parametrize("kernel", ["rbf"])
@pytest.mark.xfail(strict=True)
def test_too_many_clusters(
    n_samples, n_clusters, tol, n_features, seed, gamma, c_0, d, kernel
):
    data, _ = make_blobs(n_samples, n_features, centers=n_clusters, random_state=seed)
    lloyd = KKMeans(
        n_clusters,
        tol=tol,
        rng=seed,
        algorithm="lloyd",
        gamma=gamma,
        d=d,
        c_0=c_0,
        kernel=kernel,
    )
    elkan = KKMeans(
        n_clusters,
        tol=tol,
        rng=seed,
        algorithm="elkan",
        gamma=gamma,
        d=d,
        c_0=c_0,
        kernel=kernel,
    )
    lloyd.fit(data)
    elkan.fit(data)
    assert all(lloyd.labels_ == elkan.labels_)
    assert np.isclose(lloyd.quality_, elkan.quality_)
