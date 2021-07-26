from sklearn.linear_model import ElasticNetCV
from cvxopt import matrix, solvers
from past.utils import old_div
import numpy as np
import math


## Elastic Net implementation


def elastic_net(control, outcome, control_test):
    # control (times x paths array)
    # outcome (times x 1 array)
    model = ElasticNetCV(l1_ratio=1)
    model.fit(control, outcome)

    return model.predict(control_test), model.coef_


## Weighting with Kernel Mean Matching


def KMM(control, outcome, control_test):
    # control (times x paths array)
    # outcome (times x 1 array)
    sig2control = meddistance(control, subsample=1000)

    k = KGauss(sig2control)
    weights, _ = kernel_mean_matching(np.transpose(control), np.transpose(outcome), k)
    return np.matmul(control_test, weights / sum(weights))


## Weighting non-negative weights summin to one - original method


def SC(control, outcome, control_test):
    # control (times x paths array)
    # outcome (times x 1 array)
    import scipy.optimize
    import cvxpy as cvx

    w = cvx.Variable((control.shape[1], 1), nonneg=True)
    objective = cvx.Minimize(cvx.sum_squares(outcome - control @ w))
    constraints = [cvx.sum(w) == 1]
    prob = cvx.Problem(objective, constraints)
    # The optimal objective value is returned by prob.solve()
    result = prob.solve(verbose=False)

    return control_test @ w.value, w.value


## matrix completion


def MC_NNM(control, outcome, control_test, outcome_test):
    # control (times x paths array)
    # outcome (times x 1 array)
    from matrix_completion import svt_solve, calc_unobserved_rmse

    full_data = np.concatenate(
        [control_test, outcome_test], axis=1
    )  # outcome_test[:,np.newaxis]
    mask = np.ones(full_data.shape)
    mask[control.shape[0] :, -1] = 0

    R_hat = svt_solve(full_data, mask)
    return R_hat[:, -1]


class KGauss:
    def __init__(self, sigma2):
        assert sigma2 > 0, "sigma2 must be > 0"
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.
        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array
        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1 == d2, "Dimensions of the two inputs must be the same"
        D2 = (
            np.sum(X1 ** 2, 1)[:, np.newaxis]
            - 2 * np.dot(X1, X2.T)
            + np.sum(X2 ** 2, 1)
        )
        K = np.exp(old_div(-D2, self.sigma2))

        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...
        Parameters
        ----------
        X, Y : n x d numpy array
        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1 == n2, "Two inputs must have the same number of instances"
        assert d1 == d2, "Two inputs must have the same dimension"
        D2 = np.sum((X - Y) ** 2, 1)
        Kvec = np.exp(old_div(-D2, self.sigma2))
        return Kvec

    def __str__(self):
        return "KGauss(w2=%.3f)" % self.sigma2


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix. Useful as a heuristic for setting Gaussian kernel's width.
    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m
    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def kernel_mean_matching(X1, X2, kx, B=10, eps=None):
    """
    An implementation of Kernel Mean Matching, note that this implementation uses its own kernel parameter
    References:
    1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching."
    2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data."

    :param X1: two dimensional sample from population 1
    :param X2: two dimensional sample from population 2
    :param kern: kernel to be used, an instance of class Kernel in kernel_utils
    :param B: upperbound on the solution search space
    :param eps: normalization error
    :return: weight coefficients for instances x1 such that the distribution of weighted x1 matches x2
    """
    nx1 = X1.shape[0]
    nx2 = X2.shape[0]
    if eps == None:
        eps = B / math.sqrt(nx1)
    K = kx.eval(X1, X1)
    kappa = np.sum(kx.eval(X1, X2), axis=1) * float(nx1) / float(nx2)

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nx1)), -np.ones((1, nx1)), np.eye(nx1), -np.eye(nx1)])
    h = matrix(
        np.r_[nx1 * (1 + eps), nx1 * (eps - 1), B * np.ones((nx1,)), np.zeros((nx1,))]
    )

    solvers.options["show_progress"] = False
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol["x"])
    objective_value = sol["primal objective"] * 2 / (nx1 ** 2) + np.sum(
        kx.eval(X2, X2)
    ) / (nx2 ** 2)

    return coef, objective_value
