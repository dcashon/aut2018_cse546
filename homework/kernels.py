import numpy as np

# D. Cashon
# Collection of functions for evaluating kernel problems

def eval_kernel_sweep(data, responses, d_range, reg_range, reg_space, kernel='poly'):
    """
    Construct polynomial kernel matrix, calculate alpha, return errors for LOO CV

    Specific to current problem. Can be generalized later.

    Args:
        -data:      column (n by 1) array of data
        -reponses:  column (n by 1) array of data
        -reg:       regularization coefficient
        -d:         polynomial power, natural number
    Returns:
        -LOO CV errors as a function of lambda.
    """
    k = np.zeros((data.shape[0] - 1, data.shape[0] -1))
    fhat_x = np.zeros(data.shape[0]) # for part a) fhat(x)
    reg = np.linspace(reg_range[0],reg_range[1],reg_space)
    if kernel == 'poly':
        d = range(d_range[0], d_range[1])
    elif kernel == 'exp':
        d = np.linspace(d_range[0], d_range[1], 10)
    e = np.zeros((len(reg), len(d)))
    row = 0
    col = 0
    for reg_vals in reg:
        col = 0
        for d_vals in d:
            error = 0
            for i in range(0,data.shape[0]):
                # compute the kernel
                temp = np.array([x for x in range(0,data.shape[0]) if x != i])
                if kernel == 'poly':
                    k[:] = (1 + data[temp] @ data[temp].T) ** d_vals
                elif kernel == 'exp':
                    k[:] = np.exp(-1 * d_vals * (np.tile(data[temp], data.shape[0] - 1) - np.tile(data[temp], data.shape[0] - 1).T)**2)
                # solve for alpha_hat
                alpha_hat = np.linalg.solve(k + reg_vals * np.identity(data.shape[0] - 1), responses[temp])
                # compute the prediction error on the LOO
                error += (responses[i] - np.sum(alpha_hat * data[temp]))**2
            e[row, col] = error / data.shape[0]
            col += 1
        row += 1
    return e

def eval_kernel_spec(data, responses, d, reg, kernel='poly'):
    """
    Construct kernel matrix for given fixed hyperparameters,
    calc alpha_hat, return fhat(x) evaluated in domain [0,1]

    Specific to current problem. Can be generalized better later.

    Args:
        -data:      column (n by 1) array of data
        -reponses:  column (n by 1) array of data
        -reg:       regularization coefficient
        -d:         polynomial power, natural number
    Returns:
        -LOO CV errors as a function of lambda.
        -fhat estimate
    """
    k = np.zeros((data.shape[0], data.shape[0]))
    fhat_x = np.zeros(100) # for part a) fhat(x)
    if kernel == 'poly':
        k[:] = (1 + data @ data.T) ** d
    elif kernel == 'exp':
        k[:] = np.exp(-1 * d * (np.tile(data, data.shape[0]) - np.tile(data, data.shape[0] - 1).T)**2)
    # solve for alpha_hat
    alpha_hat = np.linalg.solve(k + reg * np.identity(data.shape[0]), responses)
    # compute fhat(x) naive
    x = np.linspace(0,1,100) # x domain arbitrary spacing
    for i in range(0, 100):
        temp_fval = 0
        for j in range(0, data.shape[0]):
            temp_fval += alpha_hat[j] * (1 + data[j] * x[i])**d
        fhat_x[i]= temp_fval
    return fhat_x