import argparse
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from inbetween.utils_approximator import load_targets


matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
font = {'family': 'cmr10', 'size': 18}
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath,amssymb}"


def load_pred(result_path):
    with open(result_path, 'rb') as f:
        data = pickle.load(f)
    return data['mean'], data['var']


def make_plots(d, savedir, inference):
    for stat in ['mean', 'var']:
        stat_results = dict()
        for i in d['results'].keys():
            stat_results[i] = d['results'][i][stat]
        bound = d['bound'] if stat == 'var' else False
        make_plot(d['X'], d[f'target_{stat}'], stat_results,
                  savedir, stat, d['depth'], inference, bound)


def make_plot(X, target_y, results_y, savedir, statistic, depth, inference, bound):
    plt.figure(figsize=(6, 3))
    plt.plot(X, target_y, label='Target')
    for i, r in results_y.items():
        if i == 'ApproximatorFFGBNN':
            color = 'C1'
            label = 'FFG'
        elif i == 'ApproximatorDropoutBNN':
            color = 'C2'
            label = 'MCDO'
        plt.plot(X, r, label=label, color=color)
        if i == 'ApproximatorFFGBNN' and bound:
            left_half_min = np.min(r[:len(r) // 2])
            right_half_min = np.min(r[len(r) // 2:])
            bound_value = left_half_min + right_half_min
            plt.hlines(bound_value, np.min(X), np.max(X), color='r',
                       linestyles='dashed', label='Bound')
    methods_name = '_'.join(inference)
    figpath = Path(savedir, f'{methods_name}_{statistic}_{depth}HL.pdf')
    plt.legend(loc=1, frameon=False, fontsize=14)
    plt.xlabel('$x$')
    if statistic == 'mean':
        plt.ylabel('$\mathbb{E}[f(x)]$')
    elif statistic == 'var':
        plt.ylabel('$\mathbb{V}[f(x)]$')
    else:
        raise NotImplementedError
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


if __name__ == '__main__':
    """Plot specified files"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', '-d', type=int, default=1,
                        help='Depth of networks to plot. Default: 1')
    parser.add_argument('--logdir', '-l', type=str, default='results',
                        help='Directory containing results. Default: results')
    parser.add_argument('--savedir', '-s', type=str, default='results',
                        help='Directory to save plots in. Default: results')
    parser.add_argument('--bound', '-b', action='store_true',
                        help='If passed, plot bound.')
    parser.add_argument('--inference_types', '-i', type=str, nargs='*',
                        default=['ApproximatorFFGBNN',
                                 'ApproximatorDropoutBNN'],
                        help="Methods to plot. Subset of "
                             "{ApproximatorFFGBNN', 'ApproximatorDropoutBNN'}: "
                             "Defaults to both.)")

    args = parser.parse_args()
    X, target_mean, target_var = load_targets()
    plotting_dict = dict(X=X, target_mean=target_mean, target_var=target_var,
                         results=dict(), depth=args.depth, bound=args.bound)
    for inference in args.inference_types:
        plotting_dict["results"][inference] = dict()
        path = Path(args.savedir, f'{inference}_{args.depth}HL.pkl')
        mean, var = load_pred(path)
        plotting_dict["results"][inference]["mean"] = mean
        plotting_dict["results"][inference]["var"] = var
    make_plots(plotting_dict, args.savedir, args.inference_types)
