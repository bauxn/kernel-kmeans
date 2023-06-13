import numpy as np
import pytest
from tests.conftest import create_labels, rng, split_integer
from sklearn.metrics.pairwise import pairwise_kernels
from elkan_utils import calc_outer_sum



@pytest.mark.parametrize("lim_upper", (1,100, 1000))
@pytest.mark.parametrize("n_clusters", (5,))
@pytest.mark.parametrize("n_features", (2,))
@pytest.mark.parametrize("size", [20, 1000])
def test_correctness_generated(lim_upper, n_clusters, n_features, size, kernel="linear"):
    data = rng.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))    
    km = pairwise_kernels(data, metric=kernel)
    for i in range(size):
        for j in range(n_clusters):
            mask = (labels == j)
            assert(np.isclose(calc_outer_sum(km, i, j, labels), km[i][mask].sum()))

#def test_correctness_manual(data): 