import argparse
import os
import pickle
import numpy as np
from pathlib import Path
import inbetween.models
from inbetween.models import GPBNN
from inbetween.likelihoods import HomoskedasticGaussianRegression
from torch.nn.functional import relu
from inbetween.utils import build_model_parameters
from inbetween.utils_2d import make_slice_plot
import matplotlib.pyplot as plt


def gen_cluster_data(input_dim, seed, sigma_w=4., sigma_b=1., depth=1,
                     noise_std=.01):
    """Generate 100 data points for regression. The inputs are located at
    two clusters on a sphere centred on the origin in input space, with radius
    sqrt(input_dim). This is to match the normalisation commonly found in
    regression, where each dimension X_i of the data is normalised such that
    E[X_i] = 0, E[X_i^2] = 1. In that case, E[||X||^2] = input_dim, so ||X||
    has a characteristic magnitude of sqrt(input_dim). The outputs are sampled
    from a GP with a ReLU BNN kernel.
    Args:
        input_dim (int): Dimensionality of input space.
        seed (int): Random seed.
        sigma_w (float): Weight prior scale for GPBNN.
        sigma_b (float): Bias prior scale for GPBNN.
        depth (int): Number of hidden layers in GPBNN.
        noise_std (float): Standard deviation of Gaussian noise added to the y
            values in the dataset.
    Returns:
        X ([100, input_dim] numpy array): Input locations generated.
        y ([100, 1] numpy array): Output values sampled from GP.
        slice_points ([500, input_dim] numpy array): Input points along the
            line joining the two clusters.
        slice_param ([500] numpy array): Parameter along the slice,
            distance travelled in the slice direction.
        unit_vector ([1, 2] numpy array): Unit vector in the direction of the
            slice.
    """
    np.random.seed(seed)
    # pick the centres of the clusters, on a circle of radius 1.
    centres = np.random.randn(2, input_dim)
    centres /= np.linalg.norm(centres, axis=-1, keepdims=True) + 1e-6
    centres *= np.sqrt(input_dim)

    # make Gaussian datapoints clustered around each centre
    X = np.random.randn(1, 50, input_dim) * 0.1 + centres[:, None, :]
    X = np.reshape(X, [100, input_dim])
    num_points = X.shape[0]

    # inputs for slice plot
    centre_difference = centres[1] - centres[0]
    cluster_distance = np.linalg.norm(centre_difference)
    unit_vector = (centre_difference / cluster_distance)[None,:]
    centre = np.mean(centres, axis=0)
    slice_param = np.linspace(-cluster_distance, cluster_distance, 500)
    slice_param = slice_param[:, None]
    slice_points = slice_param * unit_vector + centre
    model = GPBNN(input_dim=input_dim,
                  output_dim=1,
                  likelihood=HomoskedasticGaussianRegression(.01),
                  nonlinearity=relu,
                  num_layers=depth,
                  num_train=num_points,
                  sigma_w=sigma_w,
                  sigma_b=sigma_b)
    K = model.kernel(X, full_cov=True) + np.eye(num_points) * noise_std ** 2
    y = np.random.multivariate_normal(np.zeros(num_points), K)
    y = y[:, None]
    return X, y, slice_points, slice_param[:, 0], unit_vector


if __name__ == '__main__':
    """Generate random clusters regression data and train on those clusters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen',
                        action='store_true',
                        help='Generate the random clusters data.')
    parser.add_argument('--dirname',
                        type=str,
                        help='Name of folder to save data and plots.')
    parser.add_argument('--num_datasets',
                        '-n',
                        type=int,
                        default=5,
                        help='Number of random datasets.')
    parser.add_argument('--input_dim',
                        '-d',
                        type=int,
                        default=5,
                        help='Number of input dimensions.')
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
                        help='If passed input can be dropped out.')
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
                        default=.01,
                        help='Standard deviation of Gaussian likelihood. '
                             'Default .01')
    args = parser.parse_args()

    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(filepath)
    datadir = Path(dirpath, args.dirname)

    if args.gen:  # generate and save the datasets
        os.makedirs(datadir, exist_ok=True)
        for i in range(args.num_datasets):
            X, y, slice_points, slice_param, unit_vector = gen_cluster_data(
                input_dim=args.input_dim,
                seed=i,
                sigma_w=args.sigma_w,
                sigma_b=args.sigma_b,
                depth=args.layers,
                noise_std=args.noise_std)
            datapath = Path(datadir, f'data_{str(i)}.pkl')
            with open(datapath, 'wb') as handle:
                pickle.dump({'X': X,
                             'y': y,
                             'slice_points': slice_points,
                             'slice_param': slice_param,
                             'unit_vector': unit_vector},
                            handle)

    for i in range(args.num_datasets):
        datapath = Path(datadir, f'data_{str(i)}.pkl')
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
        X = data['X']
        y = data['y']
        slice_points = data['slice_points']
        slice_param = data['slice_param']
        unit_vector = data['unit_vector']
        num_train = X.shape[0]

        # build and train the models
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

        # plot predictions
        mean, std = model.pred_mean_std(slice_points)
        mean, std = mean[:, 0], std[:, 0]
        fig, ax = plt.subplots()
        make_slice_plot(ax, mean, std, slice_param, unit_vector, X, y,
                        dataset=None)
        figpath = Path(datadir, f'{args.inference}_{args.layers}HL_{i}.pdf')
        plt.tight_layout()
        plt.savefig(figpath)
        plt.close()
