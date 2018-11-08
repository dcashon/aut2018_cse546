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
    Reference:
    https://nlp.stanford.edu/IR-book/html/htmledition/k-means-1.html
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
    # set up variables
    cluster_means, f_val = [], 100
    while abs(f_val) > tol:
        f_val = 0
        # Generate k random disjoint partitions
        partitions, counts = generate_partitions(data.shape[0], k)
        for subsets in partitions:
            # calculate the centroid of current subset
            cluster_means.append(np.sum(data[subsets::], axis=0) / np.linalg.norm(np.array(subsets),2))
        # calculate objective function value
        for k in range(0, len(cluster_means)):
            for vecs in subsets[k]:
                f_val += np.linalg.norm((vecs - cluster_means[k]),2)**2
        # DEBUG
        print(f_val)





