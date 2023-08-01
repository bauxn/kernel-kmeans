import numpy as np
import pytest
from tests.pytest_utils import (
    create_labels,
    RNG,
    split_integer,
    ctrl_inner_sums,
    ctrl_mixed_sums,
    ctrl_centers_linear,
)  # pep328
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import euclidean
from KKMeans.elkan import _calc_center_dists

"""
As this is "private" function, there will not be invalid parameter tests.
"""


@pytest.mark.parametrize("lim_upper", [1, 100])
@pytest.mark.parametrize("n_clusters", [5, 50])
@pytest.mark.parametrize("n_features", [1, 2, 50])
@pytest.mark.parametrize("size", [20, 5000])
def test_linearkernel_generated(lim_upper, n_clusters, size, n_features):
    """
    tests correctness of centerdistances based on generated data

    1. generates random data and random labels (twice, new and old).
    2. build kernel matrix with sklearn (utilizing linear kernel,
        as then centers can be calculated explicitely)
    3. calculates math-terms with booleand indexing (slow!)
    4. calculates centers as average of all samples in each dimension for
        each cluster.
    5. calculates center distances
    6. asserts equality

    Parameters
    ----------
    lim_upper: int,
        highest values for generated data
    n_clusters: int
    size: int,
        amount of samples
    n_features: int
    """
    if size < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    # 1.
    data = RNG.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))
    labels_old = create_labels(split_integer(size, size // n_clusters))
    # 2.
    km = pairwise_kernels(data, metric="linear")
    # 3.
    sums_new = ctrl_inner_sums(km, labels, n_clusters)
    sums_mixed = ctrl_mixed_sums(km, labels, labels_old, n_clusters)
    sums_old = ctrl_inner_sums(km, labels_old, n_clusters)
    # 4.
    centers_new = ctrl_centers_linear(data, labels, n_clusters, n_features)
    centers_old = ctrl_centers_linear(data, labels_old, n_clusters, n_features)
    # 5.
    center_dists_control = [
        euclidean(centers_new[i], centers_old[i]) for i in range(n_clusters)
    ]
    # 6.
    sizes_new = np.array([sum(labels == i) for i in range(n_clusters)], dtype=np.int_)
    sizes_old = np.array(
        [sum(labels_old == i) for i in range(n_clusters)], dtype=np.int_
    )
    center_dists_test = _calc_center_dists(
        sums_new, sums_mixed, sums_old, sizes_new, sizes_old
    )
    assert np.allclose(center_dists_control, center_dists_test)


@pytest.mark.parametrize("lim_upper", [1, 100])
@pytest.mark.parametrize("n_clusters", [5, 50])
@pytest.mark.parametrize("n_features", [1, 2, 5, 50])
@pytest.mark.parametrize("size", [20, 500])
def test_same_centers_no_distance(lim_upper, n_clusters, size, n_features):
    """
    Tests if _calc_center_dists returns 0 when the
    centers are the same.
    """
    if size < n_clusters:
        pytest.xfail("create labels does not expect more cluster than samples")
    data = RNG.random((size, n_features)) * lim_upper
    labels = create_labels(split_integer(size, size // n_clusters))
    sums_new = np.zeros(n_clusters, dtype=np.double)
    km = pairwise_kernels(data, metric="linear")
    for i in range(n_clusters):
        mask_new = labels == i
        sums_new[i] = km[mask_new][:, mask_new].sum()
    sums_mixed = sums_new
    sums_old = sums_new
    sizes_new = np.array([sum(labels == i) for i in range(n_clusters)], dtype=np.int_)
    sizes_old = sizes_new
    center_dists_test = _calc_center_dists(
        sums_new, sums_mixed, sums_old, sizes_new, sizes_old
    )
    assert np.allclose(center_dists_test, 0.0)
