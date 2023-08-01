import numpy as np

# import pytest

SEED = 0
RNG = np.random.default_rng(SEED)


def ctrl_cluster_sizes(labels, n_clusters):
    labels = np.asarray(labels)
    return np.asarray([sum(labels == i) for i in range(n_clusters)])


def build_starting_distance(kernel_matrix, n_clusters):
    """same function as in kkmeans"""
    return np.ascontiguousarray(
        np.tile(np.diag(kernel_matrix), (n_clusters, 1)).T, dtype=np.double
    )


def create_labels(sizes, shuffled=True):
    labels = np.concatenate(
        [np.asarray([i] * sizes[i], dtype=np.int_) for i in range(len(sizes))]
    )
    if shuffled:
        RNG.shuffle(labels)
    return labels


def split_integer(integer, part_size):
    quotient, remainder = divmod(integer, part_size)
    splitted = [part_size] * quotient
    if remainder != 0:
        splitted.append(remainder)
    return np.asarray(splitted)


def ctrl_outer_sums(kernel_matrix, labels, n_clusters):
    outer_sums = np.zeros((kernel_matrix.shape[0], n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        print(mask)
        outer_sums[:, i] += kernel_matrix[:, mask].sum(axis=1)
    return outer_sums


def ctrl_inner_sums(kernel_matrix, labels, n_clusters):
    inner_sums = np.zeros(n_clusters, dtype=np.double)
    for i in range(n_clusters):
        mask = labels == i
        inner_sums[i] = kernel_matrix[mask][:, mask].sum()
    return inner_sums


def ctrl_mixed_sums(kernel_matrix, labels, labels_old, n_clusters):
    mixed_sums = np.zeros(n_clusters, dtype=np.double)
    for i in range(n_clusters):
        mask_new = labels == i
        mask_old = labels_old == i
        mixed_sums[i] = kernel_matrix[mask_new][:, mask_old].sum()
    return mixed_sums


def ctrl_centers_linear(data, labels, n_clusters, n_features):
    """helper function to calculate clusters when mapped linearly"""
    centers = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        samples_c_i = data[labels == i]
        for j in range(n_features):
            centers[i, j] = sum(samples_c_i[:, j]) / len(samples_c_i)
    return centers
