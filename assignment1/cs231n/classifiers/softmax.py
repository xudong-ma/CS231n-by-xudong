from builtins import range
import numpy as np
import cupy as cp
from random import shuffle
from past.builtins import xrange

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

    #cupy
    W = cp.array(W)
    dW = cp.array(dW)
    X = cp.array(X)
    y = cp.array(y)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    train_num = X.shape[0]
    class_num = W.shape[1]
    for i in range(train_num):
      scores = X[i].dot(W)
      scores -= cp.max(scores) #for numeric stable
      stable_scores = cp.exp(scores)
      total_stable_scores = cp.sum(stable_scores)
      loss += -cp.log(stable_scores[y[i]] / total_stable_scores)
      for j in range(class_num):
        dW[:,j] += stable_scores[j] / total_stable_scores * X[i]
        if j == y[i]:
          dW[:,j] -= X[i]

    loss /= train_num
    dW /= train_num


    loss += reg * cp.sum(W*W) #L2 regularization
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return cp.asnumpy(loss), cp.asnumpy(dW)


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #cupy
    W = cp.array(W)
    dW = cp.array(dW)
    X = cp.array(X)
    y = cp.array(y)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = cp.dot(X,W)
    scores_max = scores.max(axis = 1)[:,cp.newaxis]
    scores -=  scores_max#broadcast -
    scores = cp.exp(scores)
    total_scores = scores.sum(axis=1)[:,cp.newaxis] #broadcast /
    stable_scores = scores / total_scores
    # stable_scores = cp.asnumpy(stable_scores)
    loss = cp.mean(-cp.log(stable_scores[cp.arange(num_train),y]))
    loss += reg * cp.sum(W*W)

    stable_scores[cp.arange(num_train),y] -= 1
    dW = cp.dot(cp.transpose(X),stable_scores)
    dW /= num_train
    dW += reg * 2* W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return cp.asnumpy(loss), cp.asnumpy(dW)
