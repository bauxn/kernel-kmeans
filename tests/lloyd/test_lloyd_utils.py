import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_kernels
from lloyd_utils import _calc_update


def outer_sums(kernel_matrix, labels, n_clusters):
    outer_sums = np.zeros((kernel_matrix.shape[0], n_clusters))
    for i in range(n_clusters):
        mask = (labels == i)
        outer_sums[:, i] = kernel_matrix[:, mask].sum(axis=1)
    return outer_sums

def inner_sums(kernel_matrix, labels, n_clusters):
    inner_sums = np.zeros(n_clusters)
    for i in range(n_clusters):
        mask = (labels == i)
        inner_sums[i] = kernel_matrix[mask][:, mask].sum()
    return inner_sums

def cluster_sizes(labels, n_clusters):
    return [sum(labels == i) for i in range(n_clusters)]



@pytest.mark.parametrize("size", [1,10,100,1000])
@pytest.mark.parametrize("n_clusters", [1,2,5,20])
def test_calc_update(size, n_clusters):
    X, labels = make_blobs(size, centers=n_clusters)
    kernel = pairwise_kernels(X)
    outer, inner, sizes = _calc_update(kernel, labels, n_clusters)
    assert np.allclose(outer, outer_sums(kernel, labels, n_clusters))
    assert np.allclose(inner, inner_sums(kernel, labels, n_clusters))
    assert np.allclose(sizes, cluster_sizes(labels, n_clusters))



