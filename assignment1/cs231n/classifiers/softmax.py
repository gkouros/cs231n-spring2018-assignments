import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        #  calculate loss
        all_class_scores = X[i].dot(W)
        true_class_score = all_class_scores[y[i]]

        # use normalization trick to avoid numerical instability
        logc = -np.max(all_class_scores)

        all_class_scores_exp_sum = np.sum(np.exp(all_class_scores + logc))
        loss_i = -np.log(np.exp(true_class_score + logc) / all_class_scores_exp_sum)
        loss += loss_i

        #  calculate gradients
        for j in range(num_classes):
            false_class_score = X[i].dot(W[:,j]) + logc
            softmax = np.exp(false_class_score) / all_class_scores_exp_sum
            dW[:,j] += X[i,:] * (softmax - 1*(j==y[i]))

    # normalize
    loss /= num_train
    dW /= num_train

    # add regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_train = X.shape[0]
    num_classes = W.shape[1]

    #  calculate loss
    all_class_scores = X.dot(W)
    true_class_scores = all_class_scores[np.arange(num_train), y].reshape(-1,1)
    logc = -np.max(all_class_scores, axis=1).reshape((-1,1))
    all_class_scores_exp_sum = np.sum(np.exp(all_class_scores + logc), axis=1).reshape(-1,1)
    loss = -np.sum(np.log(np.exp(true_class_scores + logc) / all_class_scores_exp_sum))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # fully-vectorized gradient computation
    softmax = np.exp(all_class_scores+logc) / all_class_scores_exp_sum
    true_class_sub = np.zeros(softmax.shape)
    true_class_sub[np.arange(softmax.shape[0]), y] = 1
    dW +=  X.T.dot(softmax - true_class_sub)

    # semi-vectorized gradient computation
    #  for i in range(num_train):
        #  all_class_scores = X[i,:].reshape(1,-1).dot(W)
        #  logc = -np.max(all_class_scores, axis=1).reshape(-1,1)
        #  all_class_scores_exp_sum = np.sum(np.exp(all_class_scores+logc), axis=1)
        #  softmax = np.exp(all_class_scores+logc) / all_class_scores_exp_sum
        #  true_class_sub = np.zeros(softmax.shape)
        #  true_class_sub[0, y[i]] = 1
        #  dW +=  (X[i,:] * (softmax - true_class_sub).reshape(-1,1)).T

    dW /= num_train
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
