import numpy as np

# D. Cashon
# Collection of functions for evaluating kernel problems

def eval_kernel_poly(data, responses, d_range, reg_range, reg_space):
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
    reg = np.linspace(reg_range[0],reg_range[1],reg_space)
    e = np.zeros((len(reg),d_range[1] - d_range[0]))
    row = 0
    col = 0
    for reg_vals in reg:
        col = 0
        for d_vals in range(d_range[0], d_range[1]):
            error = 0
            for i in range(0,30):
                # compute the kernel
                temp = np.array([x for x in range(0,30) if x != i])
                k[:] = (1 + data[temp] @ data[temp].T) ** d_vals
                # solve for alpha_hat
                alpha_hat = np.linalg.solve(k + reg_vals * np.identity(29), responses[temp])
                # compute the prediction error on the LOO
                error += (responses[i] - np.sum(alpha_hat * data[temp]))**2
            e[row, col] = error
            col += 1
        row += 1
    return e