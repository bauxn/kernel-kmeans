import numpy as np
import pytest
from tests.conftest import create_labels, RNG, split_integer
from sklearn.metrics.pairwise import pairwise_kernels
from elkan import calc_inner_sum_mixed

@pytest.mark.parametrize("lim_upper", (1,100))
@pytest.mark.parametrize("n_clusters", (5,))
@pytest.mark.parametrize("n_features", (2,10))
@pytest.mark.parametrize("size", [20, 10000])
def test_correctness_generated(lim_upper, n_clusters, n_features, size, kernel="linear"):
    data = RNG.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))
    labels_old = create_labels(split_integer(size, size // n_clusters))
    km = pairwise_kernels(data, metric=kernel)
    inner_sums = np.zeros(n_clusters)
    mixed_sums = np.zeros(n_clusters)
    for i in range(n_clusters):
        mask_new = (labels == i)
        mask_old = (labels_old == i)
        inner_sums[i] = km[mask_new][:, mask_new].sum()
        mixed_sums[i] = km[mask_new][:, mask_old].sum()
    mixed, inner = calc_inner_sum_mixed(km, labels, labels_old, n_clusters)
    assert(np.allclose(inner, inner_sums))
    assert(np.allclose(mixed, mixed_sums))
