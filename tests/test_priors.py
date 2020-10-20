import warnings
from gpflow.kernels import RBF
from utils import MMD
from inbetween.NNKernel import ReLUKernel
from inbetween.models import FFGBNN, GPBNN
from inbetween.likelihoods import Likelihood, HomoskedasticGaussianRegression
import numpy as np
from torch.nn.functional import relu
import tensorflow as tf

warnings.simplefilter("ignore")


def two_samples(gp_prior_bias_std, gp_prior_weight_std,
                bnn_prior_bias_std, bnn_prior_weight_std,
                depth, input_points, S1, S2):
    N, D = input_points.shape

    gp = GPBNN(input_dim=D, likelihood=HomoskedasticGaussianRegression(.1),
               nonlinearity=relu, num_layers=depth, num_train=N, output_dim=1,
               sigma_w=gp_prior_weight_std, sigma_b=gp_prior_bias_std)
    gp_sample = gp.pred_sample(X_test=input_points, num_samples=S1)[:, :, 0]
    bnn = FFGBNN(input_dim=D, likelihood=Likelihood, num_train=0, output_dim=1,
                 nonlinearity=relu, num_layers=depth, width=100,
                 sigma_w=bnn_prior_weight_std, sigma_b=bnn_prior_bias_std)
    bnn_sample = bnn.sample_prior(x_data=input_points, num_samples=S2)[:, :, 0]
    return gp_sample, bnn_sample


def test_prior():
    input_dimension = 3
    num_inputs = 10
    prior_bias_stds = np.sqrt([.8, 1., 4., 5.])
    prior_weight_stds = np.sqrt([.8, 1., 4., 5.])
    depth = 3
    S1 = 7000
    S2 = 7000
    threshold = 0.0003
    mmd_kernel = RBF()
    input_points = np.random.rand(num_inputs, input_dimension)
    for prior_bias_std in prior_bias_stds:
        for prior_weight_std in prior_weight_stds:
            gp_samples, bnn_samples = two_samples(
                gp_prior_weight_std=prior_weight_std,
                gp_prior_bias_std=prior_bias_std,
                bnn_prior_weight_std=prior_weight_std,
                bnn_prior_bias_std=prior_bias_std,
                depth=depth,
                input_points=input_points, S1=S1, S2=S2)
            assert MMD(mmd_kernel, gp_samples, bnn_samples, threshold)


def test_notprior():
    input_dimension = 3
    num_inputs = 10
    gp_prior_bias_stds = np.sqrt([.8, 1., 4., 5.])
    gp_prior_weight_stds = np.sqrt([.8, 1., 2., 3.])
    bnn_prior_bias_stds = np.sqrt([2., 1., 2., 5.])
    bnn_prior_weight_stds = np.sqrt([2., 3., 2., 4.])
    depth = 3
    S1 = 7000
    S2 = 7000
    threshold = 0.0003
    mmd_kernel = RBF()
    input_points = np.random.rand(num_inputs, input_dimension)
    for gp_prior_bias_std, bnn_prior_bias_std, gp_prior_weight_std, bnn_prior_weight_std in \
            zip(gp_prior_bias_stds, bnn_prior_bias_stds, gp_prior_weight_stds,
                bnn_prior_weight_stds):
        gp_samples, bnn_samples = two_samples(
            gp_prior_weight_std=gp_prior_weight_std,
            gp_prior_bias_std=gp_prior_bias_std,
            bnn_prior_weight_std=bnn_prior_weight_std,
            bnn_prior_bias_std=bnn_prior_bias_std,
            depth=depth,
            input_points=input_points, S1=S1, S2=S2)
        assert not MMD(mmd_kernel, gp_samples, bnn_samples, threshold)


def test_kerneldiag():
    input_dimension = 5
    num_inputs = 100
    gp_prior_bias_stds = np.sqrt([.8, 1., 4., 5.])
    gp_prior_weight_stds = np.sqrt([.8, 1., 2., 3.])
    depth = 5
    input_points = np.random.rand(num_inputs, input_dimension)
    for gp_prior_bias_std in gp_prior_bias_stds:
        for gp_prior_weight_std in gp_prior_weight_stds:
            kernel = ReLUKernel(prior_bias_std=gp_prior_bias_std,
                                prior_weight_std=gp_prior_weight_std,
                                depth=depth)
            kxx = kernel(input_points)
            kxxdiag = kernel.K_diag(input_points)
            assert tf.reduce_sum(
                tf.square(tf.linalg.diag_part(kxx) - kxxdiag)) < 1e-6
