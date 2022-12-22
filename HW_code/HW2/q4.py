# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    numerators = []

    a_ii = []
    for sample in range(x_train.shape[0]):
        numerators.append(-np.linalg.norm(np.subtract(test_datum, x_train[sample])) ** 2 /(2 * tau ** 2))
    numerators = np.asarray(numerators)
    denominator = np.exp(logsumexp(numerators))
    for i in range(x_train.shape[0]):
        numerator = np.exp(numerators[i])
        a_ii.append(numerator/denominator)
    A = np.diag(a_ii)
    LHS = np.linalg.multi_dot([x_train.T, A, x_train])
    LHS += np.diag([lam]*d)
    RHS = np.linalg.multi_dot([x_train.T, A, y_train])
    w = np.linalg.solve(LHS, RHS)
    y_hat = np.matmul(test_datum.T, w)
    return y_hat


def run_validation(x, y, taus, val_frac):
    """
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """
    x_train, x_vali, y_train, y_vali = train_test_split(x, y, test_size=val_frac, random_state=0)
    tau_loss_vali = []
    for tau in taus:
        total_loss = 0
        for i in range(len(y_vali)):
            y_hat = LRLS(x_vali[i], x_train, y_train, tau)
            loss = 1/2 * (y_hat - y_vali[i]) ** 2
            total_loss += loss
        tau_loss_vali.append(total_loss/len(y_vali))
        print(total_loss / len(y_vali))
    return tau_loss_vali


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = [2]
    test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(test_losses)
    plt.show()

