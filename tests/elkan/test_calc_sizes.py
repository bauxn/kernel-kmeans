import numpy as np
import pytest
from elkan_utils import calc_sizes
from tests.conftest import create_labels

@pytest.mark.parametrize("sizes", ([0, 20, 0], [1, 0, 0, 1], [1, 2, 0, 20], [0, 1], [0, 0, 1], [0, 1, 0], [200, 777, 20000]))
def test_correctness(sizes):
    labels = create_labels(sizes)
    res = np.asarray(calc_sizes(labels, len(sizes)))
    assert all(res == sizes)




    
