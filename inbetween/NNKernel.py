import tensorflow as tf
import numpy as np
import gpflow
from gpflow.base import Parameter
from gpflow.utilities import positive


class ReLUKernel(gpflow.kernels.Kernel):
    """
    Kernel such that the mean 0 GP with the corresponding covariance function is equal in distribution
    to an infinitely wide BNN prior with mean O and "Neal scaling" on the weights. The recursive equations used
    are from https://arxiv.org/abs/1711.00165.
    """

    def __init__(self, prior_weight_std, prior_bias_std, depth):
        """

        Args:
            prior_weight_std: non-negative float or tuple
            of length depth+1 of floats, corresponding BNN has prior variance prior_weight_std / sqrt(num_inputs)
            If tuple separate standard deviation for each layer
            prior_bias_std: non-negative float or tuple
            of length depth+1 of floats, corresponding BNN has prior variance prior_bias_std
            If tuple separate standard deviation for each layer
            depth: int, number of hidden layers in corresponding BNN
        """

        super(ReLUKernel, self).__init__()
        if isinstance(prior_weight_std, float) or isinstance(prior_weight_std, int):
            prior_weight_std = prior_weight_std * np.ones(depth + 1)
        if isinstance(prior_bias_std, float) or isinstance(prior_bias_std, int):
            prior_bias_std = prior_bias_std * np.ones(depth + 1)
        assert len(prior_weight_std) == len(prior_bias_std) == depth + 1
        self.weight_variance = Parameter(prior_weight_std ** 2, transform=positive(1e-5))
        self.bias_variance = Parameter(prior_bias_std ** 2, transform=positive(1e-5))
        self.depth = depth

    def K(self, X, X2=None):
        """
        Computes covariance matrix between k(X,X2), if X2 is None computes covariance matrix k(X,X)
        Args:
            X: [N,D] float
            X2: None or [N,D] float, if None X2=X

        Returns: [N,N] matrix k(X,X2)

        """
        D = X.shape[1]  # input dimension
        jitter = 1e-15  # jitter for arccosine for numerical reasons

        if X2 is None:  # compute symmetric version
            X2 = X

        # base case for recursive formula
        Ki = self.bias_variance[0] + self.weight_variance[0] * tf.matmul(X, X2, transpose_b=True) / D
        KiX = self.bias_variance[0] + self.weight_variance[0] * tf.reduce_sum(tf.square(X), axis=1) / D
        KiX2 = self.bias_variance[0] + self.weight_variance[0] * tf.reduce_sum(tf.square(X2), axis=1) / D

        # flattened recursion
        for i in range(1, self.depth + 1):
            sqrt_term = tf.sqrt(KiX[:, None] * KiX2[None, :])  # outer product of norms
            theta = tf.acos(jitter + (1 - 2 * jitter) * Ki/sqrt_term)  # angle, 'squash' for numerical stability
            J_term = tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
            # update kernel matrices
            Ki = self.bias_variance[i] + self.weight_variance[i] / (2 * np.pi) * sqrt_term * J_term

            if i != self.depth:  # these are only needed for the recursion, don't update on last call
                KiX = self.bias_variance[i] + KiX * self.weight_variance[i] / 2.
                KiX2 = self.bias_variance[i] + KiX2 * self.weight_variance[i] / 2.
        return Ki

    def K_diag(self, X):
        """
        Computes diagonal entries of k(X,X)
        Args:
            X: [N,D] float

        Returns: [N] float. diag(k(X,X))

        """
        D = X.shape[1]  # input dimension
        KiX = self.bias_variance[0] + self.weight_variance[0] * tf.reduce_sum(tf.square(X), axis=1) / D
        for i in range(1, self.depth + 1):
            KiX = self.bias_variance[i] + KiX * self.weight_variance[i] / 2.
        return KiX
