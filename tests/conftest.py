import numpy as np
import pytest

seed = 0
rng = np.random.default_rng(seed)

def create_labels(sizes, shuffled=True):
    labels = np.concatenate([np.linspace(i, i, sizes[i], dtype=np.int_) for i in range(len(sizes))])
    if shuffled:
        rng.shuffle(labels)
    return labels

def split_integer(integer, part_size):
    quotient, remainder = divmod(integer, part_size)
    splitted = [part_size] * quotient
    if remainder != 0:
        splitted.append(remainder)
    return splitted
