import numpy as np
import pytest
from .conftest import rng, create_labels, split_integer

@pytest.mark.xfail(strict=True)
def test_rng_shuffle_difference():
    test1 = [x for x in range(10)]
    test2 = [x for x in range(10)]
    rng.shuffle(test1)
    rng.shuffle(test2)
    assert np.allclose(test1, test2)


def test_labels_difference(size=200, n_clusters=5):
    labels = create_labels(split_integer(size, size // n_clusters))
    labels_old = create_labels(split_integer(size, size // n_clusters)) 
    assert any(labels != labels_old)