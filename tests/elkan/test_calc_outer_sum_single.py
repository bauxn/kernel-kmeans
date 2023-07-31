import numpy as np
import pytest
from tests.pytest_utils import create_labels, RNG, split_integer
from sklearn.metrics.pairwise import pairwise_kernels
from KKMeans.elkan import _calc_outer_sum_single

@pytest.mark.parametrize("n_clusters", [1, 100])
@pytest.mark.parametrize("n_features", [10])
@pytest.mark.parametrize("n_samples", [1, 20, 2000])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_correct(n_clusters, n_features, n_samples, kernel):
    if n_samples < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data = RNG.random((n_samples, n_features))
    labels = create_labels(split_integer(n_samples, n_samples // n_clusters))
    index_sample = RNG.integers(n_samples)
    index_cluster = RNG.integers(n_clusters)
    km = pairwise_kernels(data, metric=kernel)
    outer_control = km[index_sample, labels==index_cluster].sum()
    outer_test = _calc_outer_sum_single(km, index_sample, index_cluster, labels)
    assert np.allclose(outer_control, outer_test)