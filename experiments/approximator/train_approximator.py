import torch
import argparse
import inbetween.models
import pickle
import os
import numpy as np

from pathlib import Path
from inbetween.utils import build_model_parameters
from inbetween.utils_approximator import load_targets


def save_pred(mean, var, inference, depth, savedir):
    os.makedirs(savedir, exist_ok=True)
    pickle_path = Path(savedir, f'{inference}_{depth}HL.pkl')
    data = dict(mean=mean, var=var)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle)


if __name__ == '__main__':
    """Train a BNN to direct approximate a target mean and variance function
    that shows in-between uncertainty."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference',
                        '-i',
                        type=str,
                        choices=['ApproximatorFFGBNN', 'ApproximatorDropoutBNN'],
                        help='Choice of approximator BNN. Default:'
                             ' ApproximatorFFGBNN',
                        default='ApproximatorFFGBNN')
    parser.add_argument('--layers',
                        type=int,
                        default=1,
                        help='Number of hidden layers. Default: 1')
    parser.add_argument('--width',
                        type=int,
                        default=50,
                        help='Number of neurons in each hidden layer. '
                             'Default: 50')
    parser.add_argument('--num_epochs',
                        '-e',
                        type=int,
                        default=1000,
                        help='Number of epochs. Default: 1000')
    parser.add_argument('--minibatch_size',
                        type=int,
                        default=200,
                        help='Minibatch size. Default: 200')
    parser.add_argument('--samples',
                        type=int,
                        default=32,
                        help='Number of MC samples during training. '
                             'Default: 32')
    parser.add_argument('--dropout_rate',
                        type=float,
                        default=.05,
                        help='Dropout rate. Default: .05')
    parser.add_argument('--dropout_bottom',
                        action='store_true',
                        help='If true, input can be dropped out. '
                             'Default: False')
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate. Default 1e-3')
    args = parser.parse_args()

    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(filepath)
    savedir = Path(dirpath, 'results')

    # set random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # load the 1D target mean and variance dataset
    X, target_mean, target_var = load_targets()
    targets = [target_mean, target_var]
    num_train = X.shape[0]

    # build and train the model
    model_parameters, training_parameters = build_model_parameters(args)
    inference = model_parameters.pop('inference_type')
    model = getattr(inbetween.models, inference)(input_dim=X.shape[-1],
                                                 output_dim=1,
                                                 num_train=num_train,
                                                 **model_parameters)
    model.train(X, targets, verbose=True, **training_parameters)

    # make and save predictions
    mean, std = model.pred_mean_std(X, num_samples=1000)
    var = std ** 2
    save_pred(mean, var, args.inference, args.layers, savedir)
