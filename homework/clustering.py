import numpy as np

# D. Cashon
# Algorithms and supporting functions for k-means clustering


def solve_kmeans(data, k, tol, freq, max_steps, method='lloyd', pp_centers=0):
    """
    Solve for k clusters using Lloyd's algorithm

    Reference:
    https://medium.com/@kshithappens/local-maxima-2abb717d6c06
    http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

    Args:
        -data:      n by d np array
        -k:         number of desired clusters
        -tol:       stopping condition based on small change in center matrix
        -freq:      how often to calculate the objective value
        -max_steps: force a max number of iterations before stopping

    Returns:
        -f_vals:    objective value at each iteration
    """
    # variables
    centers = np.zeros((k, data.shape[1]))
    centers_prior = np.zeros((k, data.shape[1])) # may not need to allocate this
    clusters, delta, iterations, f_vals = {}, 10000, 0, []
    for j in range(0,k):
        clusters[str(j)] = []
    # initialize with k random centers drawn from n, no need to worry about replacement here n >> k
    if method is 'lloyd':
        centers[:] = data[np.random.randint(0, data.shape[0], k)]
    elif method is 'pp':
        centers[:] = pp_centers
    # assign each data point to the nearest center
    while delta > tol and iterations < max_steps:
        for idx in range(0, data.shape[0]):
            d_min = 10000
            for i in range(0,k):
                # calc distance to centers
                d = np.linalg.norm(data[idx] - centers[i], 2)
                if d < d_min:
                    d_min = d
                    min_center = i
            clusters[str(min_center)].append(idx)
        centers_prior[:] = centers[:] # reset centers and recalculate
        f_val = 0
        for key in clusters.keys():
            # calc objective function
            if iterations % freq == 0: # no need to calc function value every iteration too expensive
                for vec in clusters[key]:
                    f_val += np.linalg.norm(data[vec] - centers[int(i)], 2)**2
            # if points have been assigned to cluster, do something, else dont
            if clusters[key]:
                centers[int(key)] = 0
                centers[int(key)] = np.sum(data[clusters[key]], axis=0) / len(clusters[key])
            clusters[key] = []
        delta = np.max(np.abs(centers - centers_prior))
        print(delta)
        print(f_val)
        print(iterations)
        f_vals.append(f_val)
        iterations += 1
    return f_vals, centers

def get_pp(data, k):
    """
    Implements the k-means++ algorithm to select better initial guesses for clustering

    Reference:
    https://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
    http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

    Args:
        -data:      n by d np array
        -k:         number of desired clusters
        -tol:       stopping condition based on small change in center matrix
        -freq:      how often to calculate the objective value
        -max_steps: force a max number of iterations before stopping

    Returns:
        -clusters:  a list of starting clusters chosen via kmeans++. Feed to normal k-means
    """
    centers = np.zeros((k, data.shape[1]))
    centers[0] = data[np.random.randint(0, data.shape[0])] # get the first random center
    # use partitioned uniform distribution to find probabilities
    # technically we should choose all j != the chosen centers but, n >> k so probably fine
    s = 1 # counter for rows in centers
    while s < k:
        probabilities, factor = [], 0
        for i in range(0, data.shape[0]):
            distances = []
            for j in range(0,s):
                # find min distance to each current center
                distances.append(np.linalg.norm(data[i] - centers[j], 2)**2)
            probabilities.append(min(distances))
            factor += min(distances)
        intervals = np.cumsum([x / factor for x in probabilities])
        centers[s] = data[np.searchsorted(intervals, np.random.uniform())] # chose next center
        s += 1
    return centers






