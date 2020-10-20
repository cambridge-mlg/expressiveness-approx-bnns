import argparse
import os

import numpy as np
import torch
from experiment import ActiveExperiment
from inbetween.utils import build_model_parameters

np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        '-d',
                        type=str,
                        help='Dataset on which active learning is performed, '
                             'must be recognized by Bayesian Benchmarks',
                        default='Naval')
    parser.add_argument('--inference',
                        '-i',
                        type=str,
                        choices=['GPBNN', 'FFGBNN', 'DropoutBNN'],
                        help='Method used to approximate inference in the BNN. '
                             'Default: GP',
                        default='GP')
    parser.add_argument('--layers',
                        '-l',
                        type=int,
                        default=2,
                        help='Number of hidden layers. Default: 2.')
    parser.add_argument('--acquisition_function',
                        '-a',
                        type=str,
                        choices=["max_variance", "random"],
                        default="max_variance",
                        help='Method for selecting points during active '
                             'learning. Default: max_variance')
    parser.add_argument('--active_batch_size',
                        '-abs',
                        type=int,
                        default=1,
                        help='Number of points selected in each time step of '
                             'active learning. Default: 1')
    parser.add_argument('--num_splits',
                        type=int,
                        default=1,
                        help='Number of splits of active learning to run. '
                             'Default: 1')
    parser.add_argument('--num_initial_points',
                        type=int,
                        default=5,
                        help='Number of points in active set at start of '
                             'learning. Default: 5')
    parser.add_argument('--num_steps',
                        type=int,
                        default=50,
                        help='Number of steps of active learning to run. '
                             'Default: 50')
    parser.add_argument('--logdir',
                        type=str,
                        default='logs',
                        help="Directory to which results will be saved. "
                             "Default: 'logs'")
    parser.add_argument('--width',
                        type=int,
                        default=50,
                        help='Number of neurons in each hidden layer. '
                             'Default: 50')
    parser.add_argument('--num_epochs',
                        '--ne',
                        type=int,
                        default=1000,
                        help='Number of gradient steps between active learning.'
                             ' Default: 1000')
    parser.add_argument('--samples',
                        type=int,
                        default=32,
                        help='Number of MC samples during training. '
                             'Default: 32')
    parser.add_argument('--dropout_rate',
                        type=float,
                        default=.01,
                        help='Dropout rate. Default: .01')
    parser.add_argument('--dropout_bottom',
                        action='store_true',
                        help='If passed, input can be dropped out.')
    parser.add_argument('--sigma_w',
                        type=float,
                        default=np.sqrt(2),
                        help='sigma_w. Default: 1.41')
    parser.add_argument('--sigma_b',
                        type=float,
                        default=1.,
                        help='sigma_b. Default 1.')
    parser.add_argument('--noise_std',
                        type=float,
                        default=.01,
                        help='Noise standard deviation. Default .01')
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate. Default 1e-3')
    parser.add_argument('--minibatch_size',
                        '-mb',
                        type=int,
                        default=32,
                        help='Number of points used in minibatch. Default: 32')
    args = parser.parse_args()
    experiment = ActiveExperiment(args.dataset, args.num_splits,
                                  args.num_initial_points, args.logdir,
                                  args.num_steps)
    model_parameters, training_parameters = build_model_parameters(args)
    os.makedirs(args.logdir, exist_ok=True)
    experiment.run(model_parameters=model_parameters,
                   training_parameters=training_parameters,
                   active_batch_size=args.active_batch_size,
                   acquisition_function=args.acquisition_function)
    experiment.plot_all_results()
