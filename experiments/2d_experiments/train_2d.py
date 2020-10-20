import torch
import argparse
import numpy as np
import os
from pathlib import Path

from inbetween.likelihoods import HomoskedasticGaussianRegression
import inbetween.models
from inbetween.utils import build_model_parameters
from inbetween.utils_2d import make_2d_plot, save_2d, load_X_y, get_2d_pred


if __name__ == '__main__':
    """Train a BNN on a 2D synthetic dataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        '-d',
                        type=str,
                        choices=['origin', 'axis'],
                        help='Choice of 2D synthetic dataset. Default: origin',
                        default='origin')
    parser.add_argument('--inference',
                        '-i',
                        type=str,
                        choices=['GPBNN', 'FFGBNN', 'DropoutBNN'],
                        help='Choice of inference method. Default: FFGBNN',
                        default='FFGBNN')
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
                        help='If passed, input can be dropped out.')
    parser.add_argument('--sigma_w',
                        type=float,
                        default=4.,
                        help='sigma_w. Default: 4.')
    parser.add_argument('--sigma_b',
                        type=float,
                        default=1.,
                        help='sigma_b. Default 1.')
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate. Default 1e-3')
    parser.add_argument('--noise_std',
                        type=float,
                        default=.1,
                        help='Standard deviation of Gaussian likelihood. '
                             'Default .1')
    parser.add_argument('--plot',
                        action='store_true',
                        help='Make 2d contour plot.')
    parser.add_argument('--save',
                        action='store_true',
                        help='Save the results for future plotting.')
    parser.add_argument('--savedir',
                        type=str,
                        help='Directory to save figures and data in.')
    args = parser.parse_args()

    # set random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # load the 2D synthetic dataset
    X, y = load_X_y(args.dataset)
    num_train = X.shape[0]

    # build and train the model
    model_parameters, training_parameters = build_model_parameters(args)
    inference = model_parameters.pop('inference_type')
    noise_std = model_parameters.pop("noise_std")
    likelihood = HomoskedasticGaussianRegression(noise_std=noise_std)
    model = getattr(inbetween.models, inference)(likelihood=likelihood,
                                                 input_dim=X.shape[-1],
                                                 num_train=num_train,
                                                 output_dim=y.shape[-1],
                                                 **model_parameters)
    model.train(X, y, verbose=True, **training_parameters)

    # plot and save
    contour_std, slice_mean, slice_std = get_2d_pred(model, args.dataset)
    os.makedirs(args.savedir, exist_ok=True)
    if args.plot:
        figpath = Path(args.savedir, f'{args.layers}HL_{args.inference}.pdf')
        make_2d_plot(contour_std, slice_mean, slice_std, args.dataset, figpath)
    if args.save:
        pickle_path = Path(args.savedir,
                           f'{args.layers}HL_{args.inference}.pkl')
        save_2d(contour_std, slice_mean, slice_std, pickle_path)
