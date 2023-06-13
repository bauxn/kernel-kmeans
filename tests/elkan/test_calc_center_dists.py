import numpy as np
import pytest
from tests.conftest import create_labels, RNG, split_integer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.datasets import make_blobs
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestCentroid
from elkan import calc_center_dists

'''
To see if this works correctly, there will be tests that do not only check 
if the code is doing what it should do, but also if the math behind this works.
To do so, I will manually create centers, labels and data and check if the
distances between the new and old centers are correct.
'''



@pytest.mark.parametrize("lim_upper", [1,100])
@pytest.mark.parametrize("n_clusters", [5,50])
@pytest.mark.parametrize("n_features", [1,2,5,50])
@pytest.mark.parametrize("size", [20,5000])
def test_math_linear_generated(lim_upper, n_clusters, size, n_features):
    if size < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data = RNG.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))
    labels_old = create_labels(split_integer(size, size // n_clusters))
    km = pairwise_kernels(data, metric="linear")
    sums_new = np.zeros(n_clusters, dtype=np.double)
    sums_mixed = np.zeros(n_clusters, dtype=np.double)
    sums_old = np.zeros(n_clusters, dtype=np.double)
    for i in range(n_clusters):
        mask_new = (labels == i)
        mask_old = (labels_old == i)
        sums_new[i] = km[mask_new][:, mask_new].sum()
        sums_mixed[i] = km[mask_new][:, mask_old].sum()
        sums_old[i] = km[mask_old][:, mask_old].sum()
    centers_new = calc_centers(data, labels)
    centers_old = calc_centers(data, labels_old)
    center_dists_control = [euclidean(centers_new[i], centers_old[i]) for i in range(n_clusters)]
    sizes_new = np.array([sum(labels == i) for i in range(n_clusters)], dtype=np.int_)
    sizes_old = np.array([sum(labels_old == i) for i in range(n_clusters)], dtype=np.int_)
    center_dists_test = calc_center_dists(sums_new, sums_mixed, sums_old, sizes_new, sizes_old)
    assert np.allclose(center_dists_control, center_dists_test)

@pytest.mark.parametrize("lim_upper", [1,100])
@pytest.mark.parametrize("n_clusters", [5,50])
@pytest.mark.parametrize("n_features", [1,2,5,50])
@pytest.mark.parametrize("size", [20,500])
def test_same_centers_no_distance(lim_upper, n_clusters, size, n_features):
    if size < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data = RNG.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))
    sums_new = np.zeros(n_clusters, dtype=np.double)
    km = pairwise_kernels(data, metric="linear")
    for i in range(n_clusters):
        mask_new = (labels == i)
        sums_new[i] = km[mask_new][:, mask_new].sum()
    sums_mixed = sums_new
    sums_old = sums_new
    sizes_new = np.array([sum(labels == i) for i in range(n_clusters)], dtype=np.int_)
    sizes_old = sizes_new
    center_dists_test = calc_center_dists(sums_new, sums_mixed, sums_old, sizes_new, sizes_old)
    assert np.allclose(center_dists_test, 0.)



def calc_centers(data, labels):
    clf = NearestCentroid()
    clf.fit(data, labels)
    return clf.centroids_