import numpy as np
import torch
from torch.nn.functional import relu

from utils import randomise_MFVI_net
from inbetween.models import FFGBNN, DropoutBNN
from inbetween.likelihoods import HomoskedasticGaussianRegression, Likelihood


def mfvi_ll_close(eval_batch_size):
    input_dim = 1
    output_dim = 1
    num_inputs = 2
    depth = 3
    threshold = 0.01

    X = np.random.rand(num_inputs, input_dim)
    y = np.random.rand(num_inputs, output_dim)
    likelihood = HomoskedasticGaussianRegression(.1)
    bnn = FFGBNN(input_dim=input_dim,
                 output_dim=output_dim,
                 likelihood=likelihood,
                 num_train=0,
                 nonlinearity=relu,
                 num_layers=depth,
                 width=50,
                 sigma_w=np.sqrt(2.),
                 sigma_b=1.)
    bnn.train(X, y, num_epochs=2000, batch_size=100, samples=32, lr=1e-3)

    # coherent
    c_ll = bnn.predict_log_density(X, y, local=False, num_samples=10000)
    # incoherent
    i_ll = bnn.predict_log_density(X, y, local=True, batch_size=eval_batch_size,
                                   num_samples=10000)

    return np.allclose(c_ll, i_ll, rtol=threshold)


def dropout_ll_close(eval_batch_size):
    input_dim = 1
    output_dim = 1
    num_inputs = 2
    depth = 3
    threshold = 0.01

    X = np.random.rand(num_inputs, input_dim)
    y = np.random.rand(num_inputs, output_dim)
    likelihood = HomoskedasticGaussianRegression(.1)
    bnn = DropoutBNN(input_dim=input_dim,
                     output_dim=output_dim,
                     likelihood=likelihood,
                     num_train=0,
                     nonlinearity=relu,
                     num_layers=depth,
                     width=50,
                     sigma_w=np.sqrt(2.),
                     sigma_b=1.,
                     dropout_rate=.05,
                     dropout_bottom=False)
    bnn.train(X, y, num_epochs=2000, batch_size=100, samples=32, lr=1e-3)

    # coherent
    c_ll = bnn.predict_log_density(X, y, local=False, num_samples=10000)
    # incoherent
    i_ll = bnn.predict_log_density(X, y, local=True, batch_size=eval_batch_size,
                                   num_samples=10000)

    return np.allclose(c_ll, i_ll, rtol=threshold)


def mfvi_mean_var_close(eval_batch_size):
    input_dim = 5
    output_dim = 5
    num_inputs = 10
    depth = 2
    threshold = 0.05
    num_repeats = 10000

    X = np.random.rand(num_inputs, input_dim)
    bnn = FFGBNN(input_dim=input_dim,
                 output_dim=output_dim,
                 likelihood=Likelihood,
                 num_train=0,
                 nonlinearity=relu,
                 num_layers=depth,
                 width=50,
                 sigma_w=np.sqrt(2.),
                 sigma_b=1.)
    randomise_MFVI_net(bnn)

    c_means = []
    i_means = []
    c_vars = []
    i_vars = []
    for i in range(num_repeats):
        # coherent
        c_mean, c_std = bnn.pred_mean_std(X, local=False, num_samples=10000)
        c_means.append(c_mean)
        c_vars.append(c_std ** 2)
        # incoherent
        i_mean, i_std = bnn.pred_mean_std(X, local=True,
                                          batch_size=eval_batch_size,
                                          num_samples=10000)
        i_means.append(i_mean)
        i_vars.append(i_std ** 2)

    mean_of_means_c = np.mean(np.stack(c_means), axis=0)
    mean_of_means_i = np.mean(np.stack(i_means), axis=0)
    mean_of_vars_c = np.mean(np.stack(c_vars), axis=0)
    mean_of_vars_i = np.mean(np.stack(i_vars), axis=0)

    means_close = np.allclose(mean_of_means_c.sum(),
                              mean_of_means_i.sum(), rtol=threshold)
    vars_close = np.allclose(mean_of_vars_c.sum(),
                             mean_of_vars_i.sum(), rtol=threshold)
    return means_close and vars_close


def dropout_mean_var_close(eval_batch_size):
    input_dim = 5
    output_dim = 5
    num_inputs = 10
    depth = 2
    threshold = 0.05
    num_repeats = 10000

    X = np.random.rand(num_inputs, input_dim)
    bnn = DropoutBNN(input_dim=input_dim,
                     output_dim=output_dim,
                     likelihood=Likelihood,
                     num_train=0,
                     nonlinearity=relu,
                     num_layers=depth,
                     width=50,
                     sigma_w=np.sqrt(2.),
                     sigma_b=1.,
                     dropout_rate=.05,
                     dropout_bottom=False)

    c_means = []
    i_means = []
    c_vars = []
    i_vars = []
    for i in range(num_repeats):
        # coherent
        c_mean, c_std = bnn.pred_mean_std(X, local=False, num_samples=10000)
        c_means.append(c_mean)
        c_vars.append(c_std ** 2)
        # incoherent
        i_mean, i_std = bnn.pred_mean_std(X, local=True,
                                          batch_size=eval_batch_size,
                                          num_samples=10000)
        i_means.append(i_mean)
        i_vars.append(i_std ** 2)

    mean_of_means_c = np.mean(np.stack(c_means), axis=0)
    mean_of_means_i = np.mean(np.stack(i_means), axis=0)
    mean_of_vars_c = np.mean(np.stack(c_vars), axis=0)
    mean_of_vars_i = np.mean(np.stack(i_vars), axis=0)

    means_close = np.allclose(mean_of_means_c.sum(),
                              mean_of_means_i.sum(), rtol=threshold)
    vars_close = np.allclose(mean_of_vars_c.sum(),
                             mean_of_vars_i.sum(), rtol=threshold)
    return means_close and vars_close


def test_mfvi_ll():
    """Test the MFVI predictive LL with and without
    minibatching/local reparam
    """
    all_close = True
    for eval_batch_size in [1, 2, 3]:  # under and over the test set size
        all_close = all_close and mfvi_ll_close(eval_batch_size)
    assert all_close


def test_dropout_ll():
    """Test the MCDO predictive LL with and without
    minibatching/local reparam
    """
    all_close = True
    for eval_batch_size in [1, 2, 3]:  # under and over the test set size
        all_close = all_close and dropout_ll_close(eval_batch_size)
    assert all_close


def test_mfvi_mean_var_close():
    """Test the MFVI pred mean and std with and without
    minibatching/local reparam
    """
    all_close = True
    for eval_batch_size in [5, 10, 20]:  # under and over the test set size
        all_close = all_close and mfvi_mean_var_close(eval_batch_size)
    assert all_close


def test_dropout_mean_var_close():
    """Test the MCDO pred mean and std with and without
    minibatching/local reparam
    """
    all_close = True
    for eval_batch_size in [5, 10, 20]:  # under and over the test set size
        all_close = all_close and dropout_mean_var_close(eval_batch_size)
    assert all_close
