import numpy as np
import torch
from torch.nn.functional import relu

from utils import randomise_MFVI_net
from inbetween.models import FFGBNN, DropoutBNN
from inbetween.likelihoods import Likelihood


def compare_forwards(bnn1, bnn2, X, num_batches, num_samples, threshold):
    """
    Compares if the means and variances of the output of bnn1 with local
    reparam and bnn2 without local reparam are the same.
    """
    assert bnn1.output_dim == bnn2.output_dim
    output_dim = bnn1.output_dim
    total_samples = num_batches * num_samples
    cumsum_local = np.zeros((X.shape[0], output_dim))
    cumsumsq_local = np.zeros((X.shape[0], output_dim))
    cumsum_global = np.zeros((X.shape[0], output_dim))
    cumsumsq_global = np.zeros((X.shape[0], output_dim))

    for i in range(num_batches):
        out_local = bnn1(X, num_samples, local=True)
        out_global = bnn2(X, num_samples, local=False)
        cumsum_local = cumsum_local + out_local.sum(axis=0)
        cumsumsq_local = cumsumsq_local + (out_local ** 2.).sum(axis=0)
        cumsum_global = cumsum_global + out_global.sum(axis=0)
        cumsumsq_global = cumsumsq_global + (out_global ** 2.).sum(axis=0)

    local_mean = cumsum_local / total_samples
    local_var = cumsumsq_local / total_samples - local_mean ** 2.
    local_std = np.sqrt(local_var)
    global_mean = cumsum_global / total_samples
    global_var = cumsumsq_global / total_samples - global_mean ** 2.
    global_std = np.sqrt(global_var)

    mean_diff = np.abs(local_mean - global_mean) / local_std
    std_diff = np.abs(1. - local_std / global_std)
    mean_close = np.max(mean_diff) < threshold
    std_close = np.max(std_diff) < threshold
    return mean_close and std_close


def test_mcdo_close():
    """
    Check MCDO outputs are close between local and non-local reparam for the
    same randomly initialisated BNN.
    """
    input_dim = 5
    output_dim = 5
    num_inputs = 10
    depth = 3
    threshold = 0.01
    num_batches = 100
    num_samples = 10000
    dropout_rate = 0.1
    dropout_bottom = True

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
                     dropout_rate=dropout_rate,
                     dropout_bottom=dropout_bottom)
    close = compare_forwards(bnn, bnn, X, num_batches, num_samples, threshold)
    assert close


def test_mcdo_not_close():
    """
    Check MCDO outputs are not close between local and non-local reparam for
    the different randomly initialised BNNs.
    """
    input_dim = 5
    output_dim = 5
    num_inputs = 10
    depth = 3
    threshold = 0.01
    num_batches = 100
    num_samples = 10000
    dropout_rate = 0.1
    dropout_bottom = True

    X = np.random.rand(num_inputs, input_dim)
    bnn1 = DropoutBNN(input_dim=input_dim,
                      output_dim=output_dim,
                      likelihood=Likelihood,
                      num_train=0,
                      nonlinearity=relu,
                      num_layers=depth,
                      width=50,
                      sigma_w=np.sqrt(2.),
                      sigma_b=1.,
                      dropout_rate=dropout_rate,
                      dropout_bottom=dropout_bottom)
    bnn2 = DropoutBNN(input_dim=input_dim,
                      output_dim=output_dim,
                      likelihood=Likelihood,
                      num_train=0,
                      nonlinearity=relu,
                      num_layers=depth,
                      width=50,
                      sigma_w=np.sqrt(2.),
                      sigma_b=1.,
                      dropout_rate=dropout_rate,
                      dropout_bottom=dropout_bottom)
    close = compare_forwards(bnn1, bnn2, X, num_batches, num_samples, threshold)
    assert not close


def test_mfvi_close():
    """
    Check MFVI outputs are close between local and non-local reparam for the
    same randomly initialisated BNN.
    """
    input_dim = 5
    output_dim = 5
    num_inputs = 10
    depth = 3
    threshold = 0.01
    num_batches = 100
    num_samples = 10000

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
    close = compare_forwards(bnn, bnn, X, num_batches, num_samples, threshold)
    assert close


def test_mfvi_not_close():
    """
    Check MFVI outputs are not close between local and non-local reparam for
    the different randomly initialised BNNs.
    """
    input_dim = 5
    output_dim = 5
    num_inputs = 10
    depth = 3
    threshold = 0.01
    num_batches = 100
    num_samples = 10000

    X = np.random.rand(num_inputs, input_dim)
    bnn1 = FFGBNN(input_dim=input_dim,
                  output_dim=output_dim,
                  likelihood=Likelihood,
                  num_train=0,
                  nonlinearity=relu,
                  num_layers=depth,
                  width=50,
                  sigma_w=np.sqrt(2.),
                  sigma_b=1.)
    randomise_MFVI_net(bnn1)
    bnn2 = FFGBNN(input_dim=input_dim,
                  output_dim=output_dim,
                  likelihood=Likelihood,
                  num_train=0,
                  nonlinearity=relu,
                  num_layers=depth,
                  width=50,
                  sigma_w=np.sqrt(2.),
                  sigma_b=1.)
    randomise_MFVI_net(bnn2)
    close = compare_forwards(bnn1, bnn2, X, num_batches, num_samples, threshold)
    assert not close
