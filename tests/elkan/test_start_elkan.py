import numpy as np
from KKMeans.elkan import start_elkan
from KKMeans.utils import fill_empty_clusters
import pytest
from tests.pytest_utils import (
    ctrl_centers_linear,
    RNG,
    build_starting_distance,
    ctrl_inner_sums,
    ctrl_cluster_sizes,
)
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances


@pytest.mark.parametrize("data_max", [1, 1000])
@pytest.mark.parametrize("n_samples", [10, 2000])
@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("n_clusters", [1, 50])
def test_correctness(data_max, n_samples, n_features, n_clusters):
    """tests against explicitely calculated centers, linear kernel necessary

    flop errors may occur when data_max or n_features is set big
    """
    if n_samples < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data = RNG.random((n_samples, n_features)) * data_max
    labels = RNG.integers(low=0, high=n_clusters, size=(n_samples), dtype=np.int_)
    labels, sizes = fill_empty_clusters(labels, n_clusters, rng=RNG)
    km = pairwise_kernels(data, metric="linear")
    centers = ctrl_centers_linear(data, labels, n_clusters, n_features)
    sq_dists_c = euclidean_distances(data, centers) ** 2
    sq_dists_t = build_starting_distance(km, n_clusters)

    sq_dists_t, inner_sums_t, sizes_t = start_elkan(
        sq_dists_t, km, labels, n_clusters, sizes
    )
    inner_sums_c = ctrl_inner_sums(km, labels, n_clusters)
    sizes_c = ctrl_cluster_sizes(labels, n_clusters)

    assert np.allclose(sq_dists_c, sq_dists_t)
    assert np.allclose(inner_sums_c, inner_sums_t)
    assert np.allclose(sizes_c, sizes_t)
