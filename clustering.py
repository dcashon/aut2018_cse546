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

def solve_kmeans_naive(data, k, tol, max_steps):
    """
    Naively solves the k-means problem for k clusters given n by d data
    Terrible idea
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
        cluster_means, f_val = [], 0
        # Generate k random disjoint partitions
        partitions, counts = generate_partitions(data.shape[0], k)
        for subsets in partitions:
            # calculate the centroid of current subset
            cluster_means.append(np.sum(data[subsets], axis=0) / np.linalg.norm(np.array(subsets),2))
        # calculate objective function value
        for idx in range(0, len(cluster_means)):
            for vecs in subsets:
                f_val += np.linalg.norm((data[vecs] - cluster_means[idx]),2)**2
        # DEBUG
        print(f_val)

def solve_kmeans_lloyd(data, k, tol):
    """
    Solve for k clusters using Lloyd's algorithm
    Reference:
    https://medium.com/@kshithappens/local-maxima-2abb717d6c06
    Args:
        -data: n by d np.array
        -k:    number of clusters desired
    :param data:
    :param k:
    :param tol:
    :return:
    """
    # variables
    centers = np.zeros((k, data.shape[1]))
    centers_prior = np.zeros((k, data.shape[1])) # may not need to allocate this
    clusters, delta = {}, 10000
    for j in range(0,k):
        clusters[str(j)] = []
    # initialize with k random centers drawn from n, no need to worry about replacement here n >> k
    centers[:] = data[np.random.randint(0, data.shape[0], k)]
    # assign each data point to the nearest center
    while delta > tol:
        for idx in range(0, data.shape[0]):
            d_min = 10000
            for i in range(0,k):
                # calc distance to centers
                d = np.linalg.norm(data[idx] - centers[i], 2)
                if d < d_min:
                    d_min = d
                    min_center = i
            clusters[str(min_center)].append(idx)
        # reset centers and recalculate
        centers_prior[:] = centers[:]
        centers[:] = 0
        for key in clusters.keys():
            centers[int(key)] = np.sum(data[clusters[key]], axis=0) / np.linalg.norm(np.array(clusters[key]))
            clusters[key] = []
        centers = np.nan_to_num(centers)
        delta = np.max(np.abs(centers - centers_prior))
        print(delta)
        # calculate the objective function



