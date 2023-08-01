import numpy as np
import pytest
from tests.pytest_utils import create_labels, RNG, split_integer
from sklearn.metrics.pairwise import pairwise_kernels
from tests.pytest_utils import ctrl_inner_sums, ctrl_outer_sums
from KKMeans.elkan import _calc_sums_full


@pytest.mark.parametrize("n_samples", [1, 10, 1000])
@pytest.mark.parametrize("n_clusters", [2, 100])
@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_correct_generated(n_samples, n_clusters, n_features, kernel):
    if n_samples < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data = RNG.random((n_samples, n_features))
    labels = create_labels(split_integer(n_samples, n_samples // n_clusters))
    km = pairwise_kernels(data, metric=kernel)
    inner_c = ctrl_inner_sums(km, labels, n_clusters)
    outer_c = ctrl_outer_sums(km, labels, n_clusters)
    outer_t, inner_t = _calc_sums_full(km, labels, n_clusters)
    assert np.allclose(inner_c, inner_t)
    assert np.allclose(outer_c, outer_t)
