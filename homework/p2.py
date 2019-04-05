import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from p2_functions import gen_data, solve_loss_cvx, min_loo_error_cvx
import seaborn as sns
sns.set()


# D. Cashon
# 12 1 2018
# Using CVXPY

# generate data
n = 50
x_data = np.array([(i-1) / (n-1) for i in range(1,n+1)])
y_data = np.array([gen_data(x) + np.random.standard_normal() for x in x_data])
y_data[24] = 0 # outlier

# define matrix D
D = np.zeros((n-1,n))
np.fill_diagonal(D, -1)
for i in range(n-1):
	D[i, i+1] = 1

# use cvxpy to formulate and solve kernel problem
# reshape data to column vecs
x_temp = np.reshape(x_data, (n,1))
y_temp = np.reshape(y_data, (n,1))
# set gamma and lambda param search ranges
# informed by lecture slides
gamma_range = (1, 5000)
lam_range = (10**-8, 0.01)
draws = 500
#errors, params = min_loo_error_cvx(x_temp, y_temp, lam_range, gamma_range, draws)

# chosen parameters based on couple runs of random draws:
# lambda = 0.0001, gamma = 3500
# train using these
# alpha_hat, kernel_mat = solve_loss_cvx(x_temp, y_temp, 0.0001, 3500)
# # make plot
# plt.scatter(x_temp, y_temp, label='Generated Data') # true data
# plt.plot(x_temp, kernel_mat @ alpha_hat, label='Predicted fhat(x)') # fhat(x) predicted
# plt.plot(x_temp, [gen_data(x) for x in x_temp], label='True f(x)')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('Function value')
# plt.savefig('least_squares_plot_p2.png', dpi=500, bbox_inches='tight')

# repeat for huber loss
errors, params = min_loo_error_cvx(x_temp, y_temp, lam_range, gamma_range, draws, loss='huber')
print(min(errors))
print(params[errors.index(min(errors))])
# choose lambda = 0.0001, gamma = 3500 again
# make plot

# alpha_hat, kernel_mat = solve_loss_cvx(x_temp, y_temp, 0.001, 500, loss='ls')
# alpha_hat_h, kernel_mat_h = solve_loss_cvx(x_temp, y_temp, 0.001, 500, loss='huber')
# print(alpha_hat - alpha_hat_h)

# plt.scatter(x_temp, y_temp, label='Generated Data')
# plt.scatter(x_temp, kernel_mat_h @ alpha_hat_h, label='Predicted fhat(x)')
# plt.scatter(x_temp, kernel_mat @ alpha_hat, label='ls loss predicted')
# plt.plot(x_temp, [gen_data(x) for x in x_temp], label='True f(x)')
# plt.legend()
# plt.show()

