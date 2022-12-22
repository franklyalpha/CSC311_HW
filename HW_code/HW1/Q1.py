import numpy as np
import random

import matplotlib.pyplot as plt


dimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
def generate_sample(dimension):
# generate random samples
    resulting_vec = []
    for i in range(100):
        new_vector = []
        for dim in range(dimension):
            component = random.random()
            new_vector.append(component)
        resulting_vec.append(new_vector)
    return resulting_vec


# calculate Euclidean distance for each pair of points;
def mean_and_std(resulting_vec, dimension):
    total_distance = []
    for i in range(100):
        for j in range(i + 1, 100):
            result = 0
            for k in range(dimension):
                tmp = resulting_vec[i][k] - resulting_vec[j][k]
                tmp = tmp ** 2
                result += tmp
            total_distance.append(result)
    return np.mean(total_distance), np.std(total_distance)



result_mean = []
result_std = []
for dimension in dimensions:
    samples = generate_sample(dimension)
    mean_std = mean_and_std(samples, dimension)
    result_mean.append(mean_std[0])
    result_std.append(mean_std[1])

print(result_mean)
print(result_std)

plt.plot(dimensions, result_mean)
plt.xlabel("dimensions")
plt.ylabel("mean of Euclidean distance")
plt.show()

plt.plot(dimensions, result_std)
plt.xlabel("dimensions")
plt.ylabel("std of Euclidean distance")
plt.show()
# find mean and variance
# print(total_distance.__len__())
