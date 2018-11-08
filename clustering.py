import numpy as np

# D. Cashon
# Algorithms and supporting functions for k-means clustering


def generate_partitions(num_points, num_partitions):
    """
    Generates k random disjoint subsets of range(0,num_points)
    Args:
        -num_points: n, the number of data points
        -num_partitions: the number of desired partitions of range(0,n)
    Returns:
        -partitions: a list of num_partitions lists that contain disjoint subsets of input_array
        -counts: a list of number of elements of each partition, > 0
    Warning: input_array is shuffled in place. Call sort to reorder.
    """
    to_shuffle = list(range(0, num_points))
    np.random.shuffle(to_shuffle)
    counts, partitions = [], []
    while sum(counts) != num_points:
        counts = [np.random.randint(1, num_points) for i in range(0, num_partitions)]
    for i in range(0, len(counts)):
        if i == 0:
            partitions.append(to_shuffle[0:counts[i]])
        else:
            partitions.append(to_shuffle[sum(counts[:i]): sum(counts[:i]) + counts[i]])
    return partitions, counts

def solve_naive_kmeans(data, k, tol, max_steps):
    """
    Naively solves the k-means problem for k clusters given n by d data
    Args:
        -data:      n by d (points by features) matrix of data
        -k:         number of desired clusters
        -tol:       stopping criterion
        -max_steps: max iterations before break
    Returns:

    :param data:
    :param tol:
    :param max_steps:
    :return:
    """

