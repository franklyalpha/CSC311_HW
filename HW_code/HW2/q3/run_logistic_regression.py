from HW2.q3.check_grad import check_grad
from HW2.q3.utils import *
from HW2.q3.logistic import *


import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    # below is hyperparam for large set
    # hyperparameters = {
    #     "learning_rate": 0.1,
    #     "weight_regularization": 0.1,
    #     "num_iterations": 100
    # }

   # below is for train_small
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.1,
        "num_iterations": 1000
    }

    # weights = [[0.07] for _ in range(M + 1)] # weight for large
    weights = np.array([[0.09] for _ in range(M + 1)]) # weight for small
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    train_errors = []
    val_errors = []
    # below is for training
    for t in range(hyperparameters["num_iterations"]):
        f, df, train_y = logistic(weights, train_inputs, train_targets, "s")
        update = hyperparameters["learning_rate"] * df
        weights = np.subtract(weights, update)
        train_err = evaluate(train_targets, train_y)[0]
        # validate
        val_y = logistic_predict(weights, valid_inputs)
        val_err = evaluate(valid_targets, val_y)[0]
        train_errors.append(train_err)
        val_errors.append(val_err)

    # below will plot graphs
    x_coord = [i for i in range(0, hyperparameters["num_iterations"])]
    plt.plot(x_coord, train_errors, label = "training")
    plt.plot(x_coord, val_errors, label = "validation")
    plt.legend()
    plt.show()

    #below will load test set and test
    test_input, test_target = load_test()
    y = logistic_predict(weights, test_input)
    res = evaluate(test_target, y)
    print("final train_ce: {}, final validation_ce: {}, final test_ce: {}".format(
        train_errors[-1], val_errors[-1], res
    ))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    # run_pen_logistic_regression()
