import numpy as np
import pytest
from tests.pytest_utils import (
    create_labels,
    RNG,
    split_integer,
    ctrl_inner_sums,
    ctrl_mixed_sums,
)
from sklearn.metrics.pairwise import pairwise_kernels
from KKMeans.elkan import _calc_inner_sums_mixed


@pytest.mark.parametrize("lim_upper", (1, 100))
@pytest.mark.parametrize(
    "n_clusters",
    (
        1,
        5,
    ),
)
@pytest.mark.parametrize("n_features", (2, 10))
@pytest.mark.parametrize("size", [20, 2000])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_correctness_generated(lim_upper, n_clusters, n_features, size, kernel):
    """compares to slower, simpler computed version with boolean indexing"""
    data = RNG.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))
    labels_old = create_labels(split_integer(size, size // n_clusters))
    km = pairwise_kernels(data, metric=kernel)
    inner_c = ctrl_inner_sums(km, labels, n_clusters)
    mixed_c = ctrl_mixed_sums(km, labels, labels_old, n_clusters)
    mixed_t, inner_t = _calc_inner_sums_mixed(km, labels, labels_old, n_clusters)
    assert np.allclose(inner_t, inner_c)
    assert np.allclose(mixed_t, mixed_c)
