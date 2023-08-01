import pytest
import numpy as np
from KKMeans import KKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from tests.pytest_utils import ctrl_centers_linear, RNG


@pytest.mark.parametrize("n_samples", [1, 500, 2000])
@pytest.mark.parametrize("n_predict", [1, 10, 2000])
@pytest.mark.parametrize("n_features", [1, 200])
@pytest.mark.parametrize("n_clusters", [1, 20, 100])
@pytest.mark.parametrize("seed", list(range(5)))
def test_predict_linear(n_samples, n_predict, n_features, n_clusters, seed):
    if n_samples < n_clusters:
        pytest.xfail()
    data, _ = make_blobs(n_samples, n_features, centers=n_clusters, random_state=seed)
    pred = RNG.random((n_predict, n_features))
    kkm = KKMeans(n_clusters)
    kkm.fit(data)
    centers = ctrl_centers_linear(data, kkm.labels_, n_clusters, n_features)
    preds_c = np.argmin(euclidean_distances(pred, centers), axis=1)
    preds_t = kkm.predict(pred)
    assert all(preds_c == preds_t)
