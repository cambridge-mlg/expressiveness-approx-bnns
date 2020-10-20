import numpy as np
import torch
from torch.nn import Parameter


def MMD(kernel, p_samples, q_samples, threshold):
    """
    Perform a kernel two-sample hypothesis test to test the null hypothesis P ~ Q.
    We use the unbiases estimator from https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    :param kernel: gpflow.kernel.Kernel object, determines the RKHS used in defining MMD
    :param p_samples: np.array, [S1, D] S samples from P a distribution over R^d
    :param q_samples: np.array, [S2, D] S samples from Q a distribution over R^d
    :param threshold: float, determines the threshold used for testing the null hypothesis
    :return: bool: indicator True if fail to reject null hypothesis, False if null hypothesis is rejected
    """
    S1, S2 = p_samples.shape[0], q_samples.shape[0]
    kpp, kqq, kpq = kernel(p_samples), kernel(q_samples), kernel(p_samples, q_samples)
    p_sum = 1. / (S1 * (S1-1.)) * (np.sum(kpp) - np.sum(np.diag(kpp)))
    q_sum = 1. / (S2 * (S2-1.)) * (np.sum(kqq) - np.sum(np.diag(kqq)))
    cross_sum = 1. / (S1 * S2) * np.sum(kpq)
    return p_sum + q_sum - 2. * cross_sum < threshold


def randomise_MFVI_net(bnn):
    """randomise the means and logvars in an MFVI BNN"""
    for layer in bnn.layers:
        layer.w_mean = Parameter(torch.randn_like(layer.w_mean))
        layer.w_logstd = Parameter(torch.randn_like(layer.w_logstd) / 2)
        layer.b_mean = Parameter(torch.randn_like(layer.b_mean))
        layer.b_logstd = Parameter(torch.randn_like(layer.b_logstd) / 2)
