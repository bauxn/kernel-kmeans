import pytest
import numpy as np
from KKMeans import KKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_kernels
from tests.pytest_utils import RNG


@pytest.mark.parametrize("q_metric", ["inertia", "silhouette"])
@pytest.mark.parametrize("n_samples", [1, 1000])
@pytest.mark.parametrize("n_clusters", [2, 100])
def test_measure_iter_tol0(q_metric, n_samples, n_clusters):
    dists = np.ones((n_samples, n_clusters))
    labels = RNG.integers(n_clusters, size=(n_samples), dtype=np.int_)
    labels_old = labels + 1
    quality = np.NINF
    kkm = KKMeans(n_clusters, q_metric=q_metric, tol=0)
    qual_t, conv_t = kkm._measure_iter(dists, labels, labels, quality)
    assert qual_t != np.NINF
    assert conv_t == True
    print(quality)
    qual_t, conv_t = kkm._measure_iter(dists, labels, labels_old, quality)
    assert qual_t == np.NINF
    assert conv_t == False


@pytest.mark.parametrize("q_metric", ["inertia", "silhouette"])
@pytest.mark.parametrize("n_samples", [1, 1000])
@pytest.mark.parametrize("n_clusters", [2, 100])
def test_measure_iter_tolinf(q_metric, n_samples, n_clusters):
    dists = np.ones((n_samples, n_clusters))
    labels = RNG.integers(n_clusters, size=(n_samples), dtype=np.int_)
    labels_old = labels + 1
    quality = np.NINF
    kkm = KKMeans(n_clusters, q_metric=q_metric, tol=np.inf)
    qual_t, conv_t = kkm._measure_iter(dists, labels, labels, quality)
    assert qual_t != np.NINF
    assert conv_t == True


@pytest.mark.parametrize("q_metric", ["inertia", "silhouette"])
@pytest.mark.parametrize("n_samples", [1, 1000])
@pytest.mark.parametrize("n_clusters", [2, 100])
def test_measure_iter_tolsmall(q_metric, n_samples, n_clusters):
    dists = np.ones((n_samples, n_clusters))
    labels = RNG.integers(n_clusters, size=(n_samples), dtype=np.int_)
    labels_old = labels + 1
    quality = -10
    kkm = KKMeans(n_clusters, q_metric=q_metric, tol=1e-10)

    qual_t, conv_t = kkm._measure_iter(dists, labels, labels_old, quality)
    assert conv_t == False
    assert qual_t != -10
