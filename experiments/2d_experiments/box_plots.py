import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from inbetween.utils_2d import load_2d_pred, get_slice_points, load_X_y

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
font = {'family': 'cmr10', 'size': 16}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
plt.figure(figsize=(7, 4.4))


def make_all_boxplots(logdir, savedir, hmc=False, **kwargs):
    """Make and save the figure showing overconfidence ratio boxplots for
    depths from 1 to 10 hidden layers.
    Args:
        logdir (Path object): Directory where all MFVI, MCDO and GP predictives
            were saved by train2d.py.
        savedir (Path object): directory to save the figure.
        hmc (bool): If true, add HMC to the plot. Currently, HMC is not
            implemented in this repository, so data is loaded from a previous,
            saved run.
        **kwargs:
        hmcpath1 (Path object): Path to pickle file containing HMC 1HL data.
        hmcpath2 (Path object): Path to pickle file containing HMC 2HL data.
    """
    for depth in np.arange(1, 11):
        gp_pickle = get_pickle_path(logdir, 'GPBNN', depth)
        _, _, gp_slice_std = load_2d_pred(gp_pickle)

        ffg_pickle = get_pickle_path(logdir, 'FFGBNN', depth)
        _, _, ffg_slice_std = load_2d_pred(ffg_pickle)

        dropout_pickle = get_pickle_path(logdir, 'DropoutBNN', depth)
        _, _, dropout_slice_std = load_2d_pred(dropout_pickle)

        ffg_ratios = overconfidence_ratios(gp_slice_std, ffg_slice_std)
        dropout_ratios = overconfidence_ratios(gp_slice_std, dropout_slice_std)
        make_boxplot(ffg_ratios, 'C1', position=depth)
        make_boxplot(dropout_ratios, 'C2', position=depth + 0.2)
        # HMC only run for depths 1 and 2
        if hmc and depth in [1, 2]:
            _, _, hmc_slice_std = load_2d_pred(kwargs['hmcpaths'][depth - 1],
                                               hmc=True)
            hmc_ratios = overconfidence_ratios(gp_slice_std, hmc_slice_std)
            make_boxplot(hmc_ratios, 'C0', position=depth - 0.2)

        plt.axhline(1., ls='--', color='k')  # line at ratio 1
        plt.yscale('log')
        plt.yticks([.1, 1, 10, 100])
        plt.xticks(np.arange(10) + 1, np.arange(10) + 1)
        plt.ylabel('Overconfidence ratio')
        plt.xlabel('Number of hidden layers')

        # add a legend
        if args.hmc:
            custom_lines = [Line2D([0], [0], color='C0', lw=4),
                            Line2D([0], [0], color='C1', lw=4),
                            Line2D([0], [0], color='C2', lw=4)]
            plt.legend(custom_lines, ['HMC', 'MFVI', 'MCDO'], loc=1,
                       frameon=False, ncol=3)
        else:
            custom_lines = [Line2D([0], [0], color='C1', lw=4),
                            Line2D([0], [0], color='C2', lw=4)]
            plt.legend(custom_lines, ['MFVI', 'MCDO'], loc=1,
                       frameon=False, ncol=2)
    plt.tight_layout()
    savepath = Path(savedir, 'all_box_plots.pdf')
    plt.savefig(savepath)
    plt.close()


def make_boxplot(ratios, facecolor, position):
    """Add box and whisker plot of the 300 overconfidence ratios.
    Args:
        ratios ([300] numpy array): overconfidence ratios on
            300 points in middle segment of diagonal line.
        facecolor (string): color of boxplot bar
        position (float): horizontal location of boxplot
    """
    boxplot = plt.boxplot(ratios,
                          whis=[0, 100],  # whiskers at min and max of data
                          positions=[position],
                          patch_artist=True)
    boxplot['boxes'][0].set_facecolor(facecolor)
    boxplot['medians'][0].set_color('k')


def get_pickle_path(logdir, inference, depth):
    path = f'{depth}HL_{inference}.pkl'
    return Path(logdir, path)


def overconfidence_ratios(reference_std, std):
    """Compute the overconfidence ratios along the slice connecting the data
    clusters in the 2d origin dataset. The overconfidence ratio is defined as
    the ratio of reference_std / std, where reference_std is the predictive
    standard deviation of some reference method (usually the wide-limit GP),
    and std is the predictive standard deviation of the method being evaluated
    (MFVI, MC Dropout, HMC).

    The overconfidence ratios are computed at 300 points evenly spaced between
    (-1.2, -1.2) and (1.2, 1.2) in the input space. This line covers the two
    data clusters.

    NOTE: This function assumes NUM_POINTS=500, and the 'origin'
    dataset was used, when creating reference_std and std.
    Args:
        reference_std ([NUM_POINTS] numpy array): predictive standard
            deviation of the reference method, which was saved by save_2d in
            inbetween.utils_2d.
        std ([NUM_POINTS] numpy array): predictive standard deviation of the
            evaluated method, which was saved by save_2d in inbetween.utils_2d.

    Returns:
        overconfidence_ratios ([300] numpy array): overconfidence ratios on
            300 points in middle segment of diagonal line.
    """
    confidence_ratios = reference_std / std
    assert len(confidence_ratios) == 500
    # take only the 300 points in the middle segment of the diagonal line
    confidence_ratios = confidence_ratios[100: -100]
    return confidence_ratios


def make_ratio_plot(inference, logdir, savedir):
    """Make and save a plot showing the overconfidence ratio compared to the GP
    at each point on the line joining (-1, -1) and (1, 1) in the 'origin' 2d
    dataset. Also plot a projection of the datapoints onto the slice.
    Args:
        inference (str): Inference type that is being compared to the GP
            reference. Either 'FFGBNN' or 'DropoutBNN'.
        logdir (Path object): Directory where all MFVI, MCDO and GP predictives
            were saved by train2d.py.
        savedir (Path object): directory to save the figure.
    """
    plt.figure(figsize=(7, 4.8))
    plt.yscale('log')
    # get the parameter values along the slice for the origin dataset
    _, slice_param, unit_vec = get_slice_points('origin')
    X, y = load_X_y('origin')
    linestyles = {1: 'dashed', 5: 'dotted', 9: 'dashdot'}
    for depth in [1, 5, 9]:
        gp_pickle = get_pickle_path(logdir, 'GPBNN', depth)
        _, _, gp_slice_std = load_2d_pred(gp_pickle)

        other_pickle = get_pickle_path(logdir, inference, depth)
        _, _, other_slice_std = load_2d_pred(other_pickle)

        overconfidence_ratios = gp_slice_std / other_slice_std
        plt.plot(slice_param, overconfidence_ratios, label=f'{str(depth)}HL',
                 ls=linestyles[depth])
    plt.axhline(1., ls='--', color='k')  # line at ratio 1
    plt.legend()

    # Plot the projection of the datapoints onto the slice
    X_projected = X @ np.transpose(unit_vec)
    X_projected = X_projected[:, 0]
    plt.scatter(X_projected, np.ones_like(X_projected), marker='x', color='k',
                s=150)

    savepath = Path(savedir, f'{inference}_overconfidence_ratios.pdf')
    plt.savefig(savepath)
    plt.close()


if __name__ == '__main__':
    """Generate box plots of overconfidence ratios from saved data"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',
                        type=str,
                        help='Path to logs directory containing all results')
    parser.add_argument('--hmc',
                        action='store_true',
                        help='Also include HMC plots.')
    parser.add_argument('--hmcpaths',
                        type=str,
                        nargs=2,
                        help='Paths to pickle files containing HMC 1HL and '
                             '2HL data')
    parser.add_argument('--savedir',
                        type=str,
                        help='Directory to save the figure.')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    make_all_boxplots(args.logdir, args.savedir, hmc=args.hmc,
                      hmcpaths=args.hmcpaths)
    make_ratio_plot('FFGBNN', args.logdir, args.savedir)
    make_ratio_plot('DropoutBNN', args.logdir, args.savedir)
