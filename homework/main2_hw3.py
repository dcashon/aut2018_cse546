import numpy as np

# D. Cashon
# 10 11 2018
# CSE 546
# HW3 Joke Recommender

# get the training and test data
train_data = np.loadtxt('train.txt', delimiter=',')
test_data = np.loadtxt('test.txt', delimiter=',')

# construct the rating matrix. ok if unrated joke is allocated 0
rating_matrix = np.zeros((int(np.max(train_data[:, 0])), int(np.max(train_data[:, 1]))))
#rating_matrix[:] = np.nan
for i in range(0,train_data.shape[0]):
    rating_matrix[int(train_data[i, 0]) - 1, int(train_data[i, 1]) - 1] = train_data[i, 2]

# solve for optimal v, say d = 5;
