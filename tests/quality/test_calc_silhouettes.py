import numpy as np
import pytest
from tests.pytest_utils import RNG, split_integer, create_labels
from KKMeans.quality import calc_silhouettes, avg_silhouette

# @pytest.mark.parametrize("n_samples", [10, 1000])
# @pytest.mark.parametrize("n_clusters", [pytest.param(1, marks=pytest.mark.xfail(strict=True)), 2, 200])
# def test_random(n_samples, n_clusters):
#     max_a = 10
#     min_b = 20
#     dists = np.full((n_samples, n_clusters), np.inf)
#     labels = create_labels(
#         split_integer(n_samples, n_samples // n_clusters)
#     )
#     rand_a = RNG.random((n_samples)) * max_a
#     rand_b = RNG.random((n_samples)) + min_b
#     dists[:, 0] = 0
#     dists[:, -1] = 0
#     silhouettes = calc_silhouettes(dists, labels)
#     assert np.allclose(silhouettes, (rand_b - rand_a) / rand_b)


@pytest.mark.parametrize("n_samples", [10, 1000])
@pytest.mark.parametrize(
    "n_clusters", [pytest.param(1, marks=pytest.mark.xfail(strict=True)), 2, 200]
)
def test_avg(n_samples, n_clusters):
    dists = np.full((n_samples, n_clusters), np.inf)
    labels = np.asarray([0] * n_samples)
    dists[:, 0] = 0
    dists[:, -1] = 0
    silhouettes = calc_silhouettes(dists, labels)
    avg_silh = avg_silhouette(dists, labels)
    assert np.allclose(silhouettes, avg_silh)


@pytest.mark.parametrize("n_samples", [10, 1000])
@pytest.mark.parametrize(
    "n_clusters", [pytest.param(1, marks=pytest.mark.xfail(strict=True)), 2, 200]
)
def test_zeros(n_samples, n_clusters):
    dists = np.full((n_samples, n_clusters), np.inf)
    labels = np.asarray([0] * n_samples)
    dists[:, 0] = 0
    dists[:, -1] = 0
    silhouettes = calc_silhouettes(dists, labels)
    assert np.allclose(silhouettes, 0)


@pytest.mark.parametrize("n_samples", [10, 1000])
@pytest.mark.parametrize(
    "n_clusters", [pytest.param(1, marks=pytest.mark.xfail(strict=True)), 2, 200]
)
@pytest.mark.parametrize("val_ab", [1, 200, -1])
def test_a_eq_b(n_samples, n_clusters, val_ab):
    if n_samples < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    dists = np.full((n_samples, n_clusters), np.inf)
    labels = create_labels(split_integer(n_samples, n_samples // n_clusters))
    index_a = labels
    index_b = (labels + 1) % n_clusters
    dists[list(range(n_samples)), index_a] = val_ab
    dists[list(range(n_samples)), index_b] = val_ab
    silhouettes = calc_silhouettes(dists, labels)
    assert np.allclose(silhouettes, 0)
