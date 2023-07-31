import pytest
import tests.pytest_utils as pu
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_blobs

@pytest.mark.parametrize("n_samples", [100,1000])
@pytest.mark.parametrize("n_clusters", [2,10])
@pytest.mark.parametrize("n_features", [1, 100])
def test_ctrl_centers_linear(n_samples, n_clusters, n_features):
    data, labels = make_blobs(n_samples, n_features, centers=n_clusters)
    ctrl = NearestCentroid()
    ctrl.fit(data, labels)
    assert np.allclose(pu.ctrl_centers_linear(data, labels, n_clusters, n_features), ctrl.centroids_)

def test_ctrl_cluster_sizes():
    sizes = np.asarray([10, 100, 1000, 7])
    labels = list()
    for i in range(len(sizes)):
        labels += [i] * sizes[i]
    pu.RNG.shuffle(labels)
    assert all(sizes == pu.ctrl_cluster_sizes(labels, 4))

def test_ctrl_outer_sums():
    km = [
        [0,1, 2],
        [-2, 0, 3],
        [-1, 1, 0],
    ]
    km = np.asarray(km)
    labels = np.asarray([1, 0, 1])
    outer_sums = [
        [1, 2],
        [0, 1],
        [1, -1],
    ]
    assert np.allclose(pu.ctrl_outer_sums(km, labels, 2), outer_sums)

def test_ctrl_inner_sums():
    km = [
        [0,1, 2],
        [-2, 0, 3],
        [-1, 1, 0],
    ]
    km = np.asarray(km)
    labels = [1, 0, 1]
    labels = np.asarray(labels)
    inner_sums = [0. ,1.]
    assert np.allclose(pu.ctrl_inner_sums(km, labels, 2), inner_sums)