import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_kernels 
from KKMeans.utils import _calc_outer_sums
from tests.pytest_utils import (
    ctrl_outer_sums, RNG,
    split_integer, create_labels
)

@pytest.mark.parametrize("max_data", [1,1000])
@pytest.mark.parametrize("n_samples", [1,2000])
@pytest.mark.parametrize("dim", [1,20])
@pytest.mark.parametrize("n_clusters", [1,5, 100])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_correctness(max_data, n_samples, dim, n_clusters, kernel):
    if n_samples < n_clusters:
        pytest.xfail("n_samples must be >= n_clusters")
    data = RNG.random((n_samples, dim)) * max_data
    labels = create_labels(
        split_integer(n_samples, n_samples//n_clusters))
    km = pairwise_kernels(data, metric=kernel)
    outer_t = _calc_outer_sums(km, labels, n_clusters)
    outer_c = ctrl_outer_sums(km, labels, n_clusters)
    assert np.allclose(outer_t, outer_c)

