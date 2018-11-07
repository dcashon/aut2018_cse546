import numpy as np
from mnist import MNIST


# D. Cashon
# Functions for CSE546 HW2

def create_w(d, k):
    w_hat = np.zeros(d)
    for j in range(1, d + 1):
        if j <= k:
            w_hat[j - 1] = j / k
        else:
            w_hat[j - 1] = 0
    return w_hat


def calc_max_param(data, labels):
    """
    Calculates the maximum lambda for LASSO such that w = very sparse
    :param data:
    :param labels:
    :return:
    """
    d = data.shape[1]
    n = data.shape[0]
    max_val = 0
    for k in range(0, d):
        temp = 2 * np.abs(data[:, k] @ (labels - (1 / n) * np.sum(labels)))
        if temp > max_val:
            max_val = temp
    return max_val


def solve_lasso(data, labels, param, tol, init_w):
    """
    solves lasso for given lambda
    :param data:
    :param labels:
    :param param:
    :param tol:
    :param init_w:
    :return:
    """
    n = data.shape[0]
    d = data.shape[1]
    w_k = np.zeros(d)  # pre-allocate
    w_k[:] = init_w[:]
    w_compare = np.ones(d)  # for comparison at end of k loop
    iterations, delta = 0, 1000  # dummy
    while np.max(np.abs(w_compare - w_k)) > tol:
        w_compare[:] = w_k[:]
        temp = labels - (w_k @ data.T)
        b = (1 / n) * np.sum(temp)
        for k in range(0, d):
            a_k = 2 * np.linalg.norm(data[:, k]) ** 2
            idx = [x for x in range(0, d) if x != k]
            c_k = 2 * ((labels - (np.ones(n) * b + w_k[idx] @ data.T[idx])) @ data[:, k])
            if c_k < -1 * param:
                w_k[k] = (c_k + param) / a_k
            elif abs(c_k) <= param:
                w_k[k] = 0
            elif c_k > param:
                w_k[k] = (c_k - param) / a_k
            else:
                print("Somethign went wrong with w assignment")
        print(np.max(np.abs(w_compare - w_k)))
        iterations += 1
    return w_k


def solve_regularization_path(data, responses, start_val, tol=0.005, factor=1.4, num_zeros=95, max_steps=1000,
                              give_all=True):
    """
    Solves the regularization path for LASSO
    :param data:
    :param responses:
    :param start_val:
    :param tol:
    :param factor:
    :param num_zeros:
    :param give_all:
    :return:
    """
    reg_path, nonzeros, false_zeros, pos_zeros, w_hats = [], [], [], [], []
    w_hat = np.zeros(data.shape[1])
    k = 0
    while len(np.nonzero(w_hat)[0]) < num_zeros and k < max_steps:
        print('Current Lambda' + str(start_val))
        w_hat = solve_lasso(data, responses, start_val, tol, init_w=w_hat)  # use w_hat from prior step
        w_hats.append(w_hat)
        reg_path.append(start_val)
        nonzeros.append(len(np.nonzero(w_hat)[0]))
        good_zeros = [x for x in np.nonzero(w_hat)[0] if x < 100]
        bad_zeros = [x for x in np.nonzero(w_hat)[0] if x > 100]
        if len(np.nonzero(w_hat)[0]) == 0:
            false_zeros.append(0)
        else:
            false_zeros.append(len(bad_zeros) / len(np.nonzero(w_hat)[0]))
        pos_zeros.append(len(good_zeros) / 100)
        start_val = start_val / factor
        k += 1
    if give_all is True:
        return reg_path, nonzeros, false_zeros, pos_zeros, w_hats
    else:
        return reg_path, nonzeros, w_hats


def load_dataset():
    """
    Loads in the MNIST digit image dataset
    :return:
    """
    mndata = MNIST('./data/')
    mndata.gz = True
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, labels_train, X_test, labels_test


def grad_descent_logistic(data, responses, param, step, tol, freq=1):
    n = data.shape[0]  # number of data points
    d = data.shape[1]  # number of features
    out = np.zeros((d + 1,1))  # allocate
    grad_w = np.zeros((d, 1))
    f_val, f_val_prior, k, condition = 0, 0, 0, 100
    iterations, f_vals, w_fits, b_fits = [], [], [], []
    while condition > tol:
        f_val_prior = f_val
        f_val = 0
        grad_w[:] = 0
        #     grad_w += ((1 / n) * ((1 / (1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) - 1) *
        #                responses[i] * np.reshape(data[i], (d, 1)))
        #     grad_b += ((1 / n) * ((1 / (1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) - 1) *
        #                responses[i])
        grad_w[:] = (np.reshape(np.sum(np.diag(np.reshape((1/n) * responses * (np.reciprocal(np.exp(-1 * responses *
                                    (np.ones(n) * out[-1] + out[:-1].T @ data.T)) + np.ones(n))
                 - np.ones(n)), n)) @ data, axis=0), (d,1)) + 2 * param * out[:-1])
        grad_b = np.sum((1/n) * responses * (np.reciprocal(np.exp(-1 * responses *
                                    (np.ones(n) * out[-1] + out[:-1].T @ data.T)) + np.ones(n))
                 - np.ones(n)))
        out[:-1] = out[:-1] - step * grad_w
        out[-1] = out[-1] - step * grad_b
        k += 1
        if k % freq == 0:
            for i in range(0, n):
                f_val += (1 / n) * np.log(1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))
            f_val += param * np.linalg.norm(out[:-1]) ** 2
            print("Current Function Value:" + str(f_val))
            condition = abs(f_val - f_val_prior)
            iterations.append(k)
            w_fits.append(np.copy(out[:-1]))
            b_fits.append(np.copy(out[-1]))
            f_vals.append(f_val)
    return w_fits, b_fits, iterations, f_vals


def stochastic_descent_logistic(data, responses, param, batch_size, step, tol, max_steps, ignore_tol=False, freq=1):
    d = data.shape[1]
    n = data.shape[0]
    out = np.zeros((d + 1, 1))
    out_prior = np.ones((d + 1, 1))
    f_val, f_val_prior, k, condition = 0, 0, 0, 100
    iterations, f_vals, w_fits, b_fits = [], [], [], []
    if ignore_tol is True:
        condition, tol = 1000, 0
    while condition > tol and k < max_steps:
        num = np.random.randint(0, n, size=batch_size)  # random selection
        grad_w = 0
        grad_b = 0
        f_val_prior = f_val
        f_val = 0
        out_prior[:] = out[:]
        if batch_size == 1:
            grad_w += ((1 / n) * ((1 / (1 + np.exp(-1 * responses[num] * (out[-1] + data[num] @ out[:-1])))) - 1) *
                       responses[num] * np.reshape(data[num], (d, 1))) + 2 / n * param * out[:-1]
            grad_b += ((1 / n) * ((1 / (1 + np.exp(-1 * responses[num] * (out[-1] + data[num] @ out[:-1])))) - 1) *
                       responses[num])
        else:
            for i in num:
                grad_w += ((1 / n) * ((1 / (1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) - 1) *
                           responses[i] * np.reshape(data[i], (d, 1)))
                grad_b += ((1 / n) * ((1 / (1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) - 1) *
                           responses[i])
            grad_w += (1 / n) * 2 * param * out[:-1]
        out[:-1] -= step * grad_w
        out[-1] -= step * float(grad_b)
        # eval function
        k += 1
        if ignore_tol is True:
            # for stochastic, too inefficient to calculate function value each iteration. more work needed
            # to clean this code up
            if k % freq == 0:
                for i in range(0, n):
                    f_val += (1 / n) * np.log(1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))
                f_val += param * np.linalg.norm(out[:-1]) ** 2
                print('Function Value = ' + str(f_val))
                print(k)
                iterations.append(k)
                w_fits.append(np.copy(out[:-1]))
                b_fits.append(np.copy(out[-1]))
                f_vals.append(f_val)
        else:
            for i in range(0, n):
                f_val += (1 / n) * np.log(1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))
            f_val += param * np.linalg.norm(out[:-1]) ** 2
            condition = abs(f_val - f_val_prior)
            print('Function Value = ' + str(f_val))
            print(k)
            iterations.append(k)
            w_fits.append(np.copy(out[:-1]))
            b_fits.append(np.copy(out[-1]))
            f_vals.append(f_val)

    return w_fits, b_fits, iterations, f_vals



def newton_descent(data, responses, param, step, freq, max_step):
    n, d = data.shape[0], data.shape[1]
    out = np.zeros((d + 1, 1))
    grad = np.zeros((d + 1, 1))
    hess = np.zeros((d + 1, d + 1))
    grad_w, grad_b, k, f_val = 0, 0, 0, 0
    iterations, w_fits, b_fits, f_vals = [], [], [], []
    while k < max_step:
        grad[:] = 0
        hess[:] = 0
        f_val = 0
        for i in range(0, n):
            hess[0:d, 0:d] += ((1 / n) *
                                               responses[i] ** 2 * ((1 / (
                                1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) -
                                                                    (1 / (1 + np.exp(
                                                                        -1 * responses[i] * (out[-1] + data[i] @ out[
                                                                                                                 :-1])))) ** 2)
                                               * np.reshape(data[i], (d, 1)) @ np.reshape(data[i], (1, d)))
            hess[d, 0:d] += (1 / n) * responses[i] ** 2 * (
                    (1 / (1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) -
                    (1 / (1 + np.exp(
                        -1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) ** 2) * np.reshape(data[i], d)
            hess[d, d] += (1 / n) * responses[i] ** 2 * (
                    (1 / (1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) -
                    (1 / (1 + np.exp(
                        -1 * responses[i] * (out[-1] + data[i] @ out[:-1])))) ** 2)

        grad[:-1] = (np.reshape(np.sum(np.diag(np.reshape((1/n) * responses * (np.reciprocal(np.exp(-1 * responses *
                                (np.ones(n) * out[-1] + out[:-1].T @ data.T)) + np.ones(n))
             - np.ones(n)), n)) @ data, axis=0), (d,1)) + 2 * param * out[:-1])
        grad[-1] = np.sum((1/n) * responses * (np.reciprocal(np.exp(-1 * responses *
                                (np.ones(n) * out[-1] + out[:-1].T @ data.T)) + np.ones(n))
             - np.ones(n)))
        hess[0:d, 0:d] += 2 * param * np.identity(d)
        hess[0:d, d] = np.reshape(hess[d, 0:d], d)
        out[:] -= step * np.linalg.solve(hess, grad)
        k += 1
        if k % freq == 0:
            for i in range(0, n):
                f_val += (1 / n) * np.log(1 + np.exp(-1 * responses[i] * (out[-1] + data[i] @ out[:-1])))
            f_val += param * np.linalg.norm(out[:-1]) ** 2
            print('Function Value = ' + str(f_val))
            print(k)
            iterations.append(k)
            w_fits.append(np.copy(out[:-1]))
            b_fits.append(np.copy(out[-1]))
            f_vals.append(f_val)
    return w_fits, b_fits, iterations, f_vals


def calc_ui(x, y, w, b):
    return 1 / (1 + np.exp(y * (b + x @ w)))


def find_fit_error(true_data, true_responses, w_fits, b_fits):
    error = []
    for j in range(0, len(w_fits)):
        pred_labels = []
        for i in range(0, true_data.shape[0]):
            # classify gradient descent
            if np.sign(b_fits[j] + true_data[i] @ w_fits[j]) > 0:
                pred_labels.append(1)  # 7
            else:
                pred_labels.append(-1)  # 2
        error.append(1 - (np.count_nonzero(true_responses == pred_labels) / true_responses.shape[0]))
    return error


def predict_error(data, w, true_responses):
    predictions = data @ w
    error = np.sum((predictions - true_responses) ** 2)
    return error

