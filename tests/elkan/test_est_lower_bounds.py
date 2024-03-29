import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.datasets import make_blobs
from tests.pytest_utils import (
    RNG,
    build_starting_distance,
    create_labels,
    split_integer,
    ctrl_cluster_sizes,
)
from KKMeans.elkan import start_elkan, _est_lower_bounds
from KKMeans.utils import calc_sizes


@pytest.mark.parametrize("max_data", [1, 1000])
@pytest.mark.parametrize("n_samples", [1, 2000])
@pytest.mark.parametrize("n_clusters", [1, 20])
@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_recalc(max_data, n_samples, n_clusters, n_features, kernel):
    """
    tests if l_bounds are updated correctly

    forces _est_lower_bounds to recompute every distance,
    check if every distance updated correctly

    1. generates two sets of labels
    2. calculates exact dists to first set
    3. set center dists to inf
    4. compute exact dists to second set
    5. compute lower bounds to second set
    6. assert that all distances are equal
    """

    if n_samples < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    X = RNG.random((n_samples, n_features)) * max_data
    labels_0 = create_labels(split_integer(n_samples, n_samples // n_clusters))
    labels_1 = RNG.permutation(labels_0)
    sizes_0 = calc_sizes(labels_0, n_clusters)
    sizes_1 = calc_sizes(labels_1, n_clusters)
    km = pairwise_kernels(X, metric=kernel)  # TODO
    start_dists = build_starting_distance(km, n_clusters)
    sq_dists, inner, __ = start_elkan(start_dists, km, labels_0, n_clusters, sizes_0)
    c_dists = np.asarray([[np.inf] * n_clusters] * n_samples, dtype=np.double)
    sq_dists_c, inner_sums, cl_sizes = start_elkan(
        start_dists, km, labels_1, n_clusters, sizes_1
    )
    sq_dists_t = _est_lower_bounds(km, sq_dists, c_dists, labels_1, cl_sizes, inner)

    assert np.allclose(sq_dists_c, sq_dists_t)


@pytest.mark.parametrize("n_samples", [1, 2000])
@pytest.mark.parametrize("n_clusters", [1, 20])
@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_skip(n_samples, n_features, n_clusters, kernel):
    if n_samples < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data, labels = make_blobs(n_samples, n_features, centers=n_clusters)
    km = pairwise_kernels(data, metric=kernel)
    c_dists = np.asarray([[-np.inf] * n_clusters] * n_samples, dtype=np.double)
    sizes = calc_sizes(labels, n_clusters)
    start_dists = build_starting_distance(km, n_clusters)
    exact_sq_dists, inner, sizes = start_elkan(
        start_dists, km, labels, n_clusters, sizes
    )
    unlikely_val = 0.00058
    l_bounds = np.asarray([[unlikely_val] * n_clusters] * n_samples, dtype=np.double)
    l_bounds[range(n_samples), labels] = exact_sq_dists[range(n_samples), labels]
    l_bounds_c = l_bounds.copy()
    l_bounds_t = np.asarray(
        _est_lower_bounds(km, l_bounds, c_dists, labels, sizes, inner)
    )
    assert np.allclose(l_bounds_c, l_bounds_t)
