import pytest
import numpy as np
from KKMeans.utils import fill_empty_clusters
from tests.pytest_utils import split_integer, create_labels, ctrl_cluster_sizes


@pytest.mark.parametrize("n_clusters", [1, 500, 1000])
@pytest.mark.parametrize("n_samples", [1, 2000, 4500])
def test_one_not_empty_fill(n_clusters, n_samples):
    if n_samples < n_clusters:
        pytest.xfail("n_samples must be >= n_clusters")
    sizes = [n_samples] + [0] * (n_clusters - 1)
    labels = create_labels(sizes)
    labels, sizes_t = fill_empty_clusters(labels, n_clusters)
    sizes_c = ctrl_cluster_sizes(labels, n_clusters)
    assert np.allclose(sizes_c, sizes_t)
    assert sizes_t.sum() == n_samples
    assert all(sizes_t != 0)


@pytest.mark.parametrize("n_clusters", [1, 500, 1000])
@pytest.mark.parametrize("n_samples", [1, 2000, 4500])
def test_one_not_empty_sizes(n_clusters, n_samples):
    if n_samples < n_clusters:
        pytest.xfail("n_samples must be >= n_clusters")
    sizes = [n_samples] + [0] * (n_clusters - 1)
    labels = create_labels(sizes)
    labels, sizes_t = fill_empty_clusters(labels, n_clusters)
    assert sizes_t.sum() == n_samples
    assert all(sizes_t != 0)


@pytest.mark.parametrize("n_clusters", [1, 500, 1000])
@pytest.mark.parametrize("n_samples", [1, 2000, 4500])
def test_none_empty_sizes(n_clusters, n_samples):
    if n_samples < n_clusters:
        pytest.xfail("n_samples must be >= n_clusters")
    sizes = [n_samples] + [0] * (n_clusters - 1)
    labels = create_labels(sizes)
    labels, sizes_t = fill_empty_clusters(labels, n_clusters)
    sizes_c = ctrl_cluster_sizes(labels, n_clusters)
    assert np.allclose(sizes_c, sizes_t)
    assert sizes_t.sum() == n_samples
    assert all(sizes_t != 0)
