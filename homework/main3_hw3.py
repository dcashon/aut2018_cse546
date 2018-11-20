import numpy as np
from homework.kernels import eval_kernel_sweep, eval_kernel_spec
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set()

# generate the data, once generated, load it back in
f = lambda x: 4 * np.sin(np.pi*x) * np.cos(np.pi * 6 * x ** 2)
# data = np.array([np.random.uniform() for x in range(0,30)])
# responses = f(data)
# pickle.dump(data, open('p2_data.pkl', 'wb'))
# pickle.dump(responses, open('p2_responses.pkl', 'wb'))
data = pickle.load(open('p2_data.pkl', 'rb'))
responses = pickle.load(open('p2_responses.pkl', 'rb'))





# create plot with the polynomial kernel
l1 = 0.01
l2 = 0.1
d1 = 3
d2 = 12
out_poly = eval_kernel_sweep(np.reshape(data,(30,1)), np.reshape(responses, (30,1)), (d1,d2), (l1, l2), 100, kernel='poly')

plt.figure(1)
for i in range(0, out_poly.shape[1]):
    plt.plot(np.linspace(l1,l2,100), out_poly[:,i], label=str(i))
plt.legend()
plt.ylabel('LOO Error (1/n)(MSE)')
plt.xlabel('Lambda')

# plot the original data, true f(x), fhat(x) for polynomial kernel
plt.figure(3)
plt.scatter(data, responses, label='Generated Data')
plt.plot(np.linspace(0,1,100), f(np.linspace(0,1,100)), label='True f(x)')
plt.plot(np.linspace(0,1, 100), eval_kernel_spec(data, responses, 10, 0.01))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

# # create plot with the exp kernel
# l1 = 0.01
# l2 = 0.1
# d1 = 3
# d2 = 5
# out_exp = eval_kernel(np.reshape(data,(30,1)), np.reshape(responses, (30,1)), (d1,d2), (l1, l2), 100, kernel='exp')
#
# plt.figure(2)
# for i in range(0, out_poly.shape[1]):
#     plt.plot(np.linspace(l1,l2,100), out_exp[:,i], label=str(i))
# plt.legend()
# plt.ylabel('LOO Error (1/n)(MSE)')
# plt.xlabel('Lambda')
#
# plt.show()
