'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
from sklearn.metrics import  accuracy_score
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    N, K = train_data.shape
    classified_data = [[] for _ in range(10)]
    # categorizing data into 10 different classes
    for i in range(N):
        curr_category = int(train_labels[i])
        curr_data = train_data[i]
        classified_data[curr_category].append(curr_data)

    means = np.zeros((10, 64))
    for classification in range(10):
        classified_data_matrix = np.asmatrix(classified_data[classification])
        mean_for_cur_class = classified_data_matrix.mean(0)
        means[classification] = mean_for_cur_class

    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    N, K = train_data.shape
    classified_data = [[] for _ in range(10)]
    # categorizing data into 10 different classes
    for i in range(N):
        curr_category = int(train_labels[i])
        curr_data = train_data[i]
        classified_data[curr_category].append(curr_data)

    for x in range(10):
        # cov = np.zeros((64, 64))
        # a = means - np.ones((10, 10)).dot(means) * 1/10
        # V = a.T.dot(a)/N
        V = np.cov(np.asmatrix(classified_data[x]).T)

        cov = V + np.diag([0.01] * V.shape[0])
        covariances[x] = cov

    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    N, D = means.shape
    K = 10
    n = len(digits)
    results = np.zeros((n, 10))
    for i in range(n):
        log_likelihood_per_sample = []  # 1 * 10
        for u in range(K):
            part_11 = (2 * np.pi) ** (-D/2)
            part_12 = np.linalg.det(covariances[u])
            part_1 = part_11 * part_12 ** (-1 / 2)

            part_2 = np.exp(-1/2 * np.linalg.multi_dot([(digits[i] - means[u]).T,covariances[u], digits[i] - means[u]]))
            result_per_class = np.log(part_1 * part_2)
            log_likelihood_per_sample.append(result_per_class)
        results[i] = np.asarray(log_likelihood_per_sample)
    return results


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    cov = generative_likelihood(digits, means, covariances)
    return cov * 1/10


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.mean(cond_likelihood, axis=0)


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    N, K = cond_likelihood.shape
    result_target = np.zeros((N, 1))
    for i in range(N):
        highest_index_per_row = np.where(cond_likelihood[i] == np.amax(cond_likelihood[i]))
        result_target[i] = highest_index_per_row
    return result_target
    # Compute and return the most likely class


def calculate_accuracy(true_test_label, generated_test_label):
    accuracy = accuracy_score(true_test_label, generated_test_label)
    return accuracy


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    generative_likelihood(train_data, means, covariances)
    # Evaluation
    generated_train_labels = classify_data(train_data, means, covariances)
    train_accuracy = calculate_accuracy(train_labels, generated_train_labels)

    generated_test_labels = classify_data(test_data, means, covariances)
    test_accuracy = calculate_accuracy(test_labels, generated_test_labels)
    print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")


if __name__ == '__main__':
    main()
