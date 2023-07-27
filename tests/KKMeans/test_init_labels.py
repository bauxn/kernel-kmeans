import pytest
import numpy as np
from KKMeans import KKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_kernels

@pytest.mark.parametrize("n_samples", [1,80,2000])
@pytest.mark.parametrize("init", ["kmeans++", "random", "truerandom", "centers", pytest.param("notimplemented", marks=pytest.mark.xfail(strict=True))])
@pytest.mark.parametrize("n_features", [1,200])
@pytest.mark.parametrize("n_clusters", [1,20])
def test_init_labels(n_samples, init, n_features, n_clusters):
    if n_samples < n_clusters:
        pytest.xfail()
    data, _, centers = make_blobs(n_samples, n_features, centers=n_clusters, return_centers=True)
    if init == "centers":
        init = centers
    kkm = KKMeans(n_clusters, init=init)
    kmatrix = pairwise_kernels(data)
    labels = kkm._init_labels(data, kmatrix)
    assert isinstance(labels, np.ndarray)