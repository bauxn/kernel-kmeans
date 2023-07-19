import pytest
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kernel_kmeans import KKMeans
from tests.pytest_utils import ctrl_centers_linear
from sklearn.metrics.pairwise import pairwise_distances
from utils import fill_empty_clusters


@pytest.mark.parametrize("n_samples", [3000])
@pytest.mark.parametrize("n_features", [1, 10, 30])
@pytest.mark.parametrize("n_clusters", [5])
@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("alg", ["lloyd", "elkan"])
def test_with_random_centers_low_nclusters(n_samples, n_features, n_clusters, seed, alg):
    '''
    Tests with given centers (that are not 100% exact, see make_blobs)

    IF FAILED, LOOK IF WARNING IN STDOUT!
    This test _might_ fail because KKMeans assigns random element to 
    an empty cluster and KMeans can therefore yield different results!
    Therefore, ratio n_samples:n_clusters needs to be huge!
    (this still is no guarantee)
    '''
    if n_samples < n_clusters:
        pytest.xfail()
    rng = np.random.default_rng(seed)

    data, _, centers= make_blobs(
        n_samples, n_features, centers=n_clusters,
        random_state=seed, return_centers=True) 
    centers = rng.choice(data, n_clusters)
    kmeans = KMeans(n_clusters, n_init=1, init=centers, random_state=seed,tol=0)
    kkmeans = KKMeans(n_clusters, n_init=1, init=centers, rng=seed, algorithm=alg, tol=0)
    kmeans.fit(data),
    kkmeans.fit(data)
    assert np.all(kmeans.labels_ == kkmeans.labels_)
    assert np.isclose(kmeans.inertia_, kkmeans.quality_)

@pytest.mark.parametrize("n_samples", [3000])
@pytest.mark.parametrize("n_features", [1, 10, 30])
@pytest.mark.parametrize("n_clusters", [5])
@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("alg", ["lloyd", "elkan"])
def test_with_good_centers_low_nclusters(n_samples, n_features, n_clusters, seed, alg):
    '''
    Tests with given centers (that are not 100% exact, see make_blobs)

    This test _might_ fail because KKMeans assigns random element to 
    an empty cluster and KMeans can therefore yield different results!
    Therefore, ratio n_samples:n_clusters needs to be huge
    '''
    if n_samples < n_clusters:
        pytest.xfail()

    data, _, centers= make_blobs(
        n_samples, n_features, centers=n_clusters,
        random_state=seed, return_centers=True) 
    kmeans = KMeans(n_clusters, n_init=1, init=centers, random_state=seed,tol=0)
    kkmeans = KKMeans(n_clusters, n_init=1, init=centers, rng=seed, algorithm=alg, tol=0)
    kmeans.fit(data),
    kkmeans.fit(data)
    assert np.all(kmeans.labels_ == kkmeans.labels_)
    assert np.isclose(kmeans.inertia_, kkmeans.quality_)

def dummy_kmeans(data,centers, n_clusters, n_features, max_iter, seed):
    gen = np.random.default_rng(seed)
    for _ in range(max_iter):
        dists = pairwise_distances(data, centers)
        labels = np.asarray(np.argmin(dists, axis=1),dtype=np.int_)
        labels, _ = fill_empty_clusters(labels, n_clusters, rng=gen)
        centers = ctrl_centers_linear(data, labels, n_clusters, n_features)
    
    return labels

@pytest.mark.parametrize("n_samples", [10, 200])
@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("n_clusters", [5, 100])
@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.parametrize("alg", ["lloyd", "elkan"])
def test_with_centers_empty_clusters(n_samples, n_features, n_clusters, seed, alg):
    '''
    tests against dummy kmeans as sklearn KMeans uses different method 
    when encountering empty clusters'''
    if n_samples < n_clusters:
        pytest.xfail()

    data, _, centers= make_blobs(
        n_samples, n_features, centers=n_clusters,
        random_state=seed, return_centers=True) 
    labels_c = dummy_kmeans(data, centers, n_clusters, n_features, 100, seed)
    kkmeans = KKMeans(n_clusters, n_init=1, init=centers, rng=seed, algorithm=alg, tol=0, max_iter=100)
    kkmeans.fit(data)
    assert np.all(labels_c == kkmeans.labels_)
