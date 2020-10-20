# show tsne representation of points chosen by active learning
import pickle
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
font = {'family': 'cmr10', 'size': 18}
matplotlib.rc('font', **font)
# matplotlib.rc('text', usetex=True)


def plot_tsne(X, X_2d, results, chosen_indices, figpath, colours='norm'):
    if colours == 'norm':
        colours = np.sqrt(np.sum(X ** 2, axis=1))
        cbar_label = 'Distance from origin'
    elif colours == 'uncertainty':
        f_std_train = results["f_std_train"].squeeze()
        f_std_test = results["f_std_test"].squeeze()
        f_std = np.concatenate([f_std_train, f_std_test], axis=0)
        colours = f_std
        cbar_label = 'Predictive standard deviation'
    else:
        raise NotImplementedError
    cm = plt.cm.get_cmap('viridis')
    plt.scatter(X_2d[:, 0], X_2d[:, 1],
                c=colours, cmap=cm, s=3, alpha=1, linewidth=0.0)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    p1 = plt.scatter(X_2d[chosen_indices, 0], X_2d[chosen_indices, 1],
                     s=60, c='r', alpha=1, label='Selected', marker='x')
    p2 = plt.scatter(X_2d[results["init_indices"], 0],
                     X_2d[results["init_indices"], 1],
                     s=60, c='silver', alpha=1, label='Initial', marker='x')
    plt.tick_params(
        axis='both',  # changes apply to both axes
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off
    #  plt.legend([p2, p1], ['Initial', 'Selected'], ncol=2)
    plt.tight_layout()
    plt.savefig(figpath, dpi=180)
    plt.close()


def make_tsne_plots(X, X_2d, results, dirpath, iter, name):
    before_indices = np.setdiff1d(results["active_indices"],
                                  results["init_indices"])
    after_indices = np.concatenate([before_indices,
                                    results['new_active_indices']])
    # concatenating with empty list can cast to float
    after_indices = [int(i) for i in after_indices]

    unc_before = f'_tsne_uncertainty_iter_{str(iter)}_before.png'
    unc_before = name + unc_before
    unc_beforepath = Path(dirpath, unc_before)
    plot_tsne(X, X_2d, results, before_indices, unc_beforepath, colours='uncertainty')

    unc_after = f'_tsne_uncertainty_iter_{str(iter)}_after.png'
    unc_after = name + unc_after
    unc_afterpath = Path(dirpath, unc_after)
    plot_tsne(X, X_2d, results, after_indices, unc_afterpath, colours='uncertainty')

    norm_before = f'_tsne_norm_iter_{str(iter)}_before.png'
    norm_before = name + norm_before
    norm_beforepath = Path(dirpath, norm_before)
    plot_tsne(X, X_2d, results, before_indices, norm_beforepath, colours='norm')

    norm_after = f'_tsne_norm_iter_{str(iter)}_after.png'
    norm_after = name + norm_after
    norm_afterpath = Path(dirpath, norm_after)
    plot_tsne(X, X_2d, results, after_indices, norm_afterpath, colours='norm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath',
                        '-d',
                        type=str,
                        help='Folder of the split to plot tsne for')
    parser.add_argument('--iter',
                        type=int,
                        help='Iteration of active learning to plot')
    parser.add_argument('--run-tsne', dest='tsne', action='store_true',
                        help='Run tsne and save the result in tsne_dirpath.'
                        ' If True, no plotting is done.')
    parser.add_argument('--tsne_dirpath',
                        '-t',
                        type=str,
                        help='Folder that contains the tsne.pkl file. If '
                             'run_tsne is True, a tsne.pkl file is created '
                             'in tsne_dirpath.')
    parser.add_argument('--savedir',
                        type=str,
                        help='Directory to save the figures.')
    parser.add_argument('--depth',
                        type=int,
                        help='Depth. Used to name figure.')
    parser.add_argument('--inference',
                        type=str,
                        help='Inference type. Used to name figure.')
    parser.add_argument('--acquisition',
                        '-a',
                        type=str,
                        help='Acquisition function. Used to name figure.')
    parser.set_defaults(tsne=False)
    args = parser.parse_args()

    # load dataset
    datapath = Path(args.dirpath, 'data.pkl')
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    X_train = data["train_data"][0]
    X_test = data["test_data"][0]
    X = np.concatenate([X_train, X_test], axis=0)
    if args.tsne:
        # perform tsne
        tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000,
                    verbose=1)
        X_2d = tsne.fit_transform(X)
        os.makedirs(args.tsne_dirpath, exist_ok=True)
        tsnepath = Path(args.tsne_dirpath, 'tsne.pkl')
        with open(tsnepath, 'wb') as f:
            pickle.dump(X_2d, f)
    else:
        # load tsne data
        tsnepath = Path(args.tsne_dirpath, 'tsne.pkl')
        with open(tsnepath, 'rb') as f:
            X_2d = pickle.load(f)

    if not args.tsne: # only plot if not generating the tsne file
        # load results
        relativepath = Path('logged-data', f'{args.iter}.pkl')
        resultspath = Path(args.dirpath, relativepath)
        with open(resultspath, 'rb') as f:
            results = pickle.load(f)
        os.makedirs(args.savedir, exist_ok=True)
        name = f'{args.inference}_{args.depth}HL_{args.acquisition}'
        make_tsne_plots(X, X_2d, results, args.savedir, args.iter, name)

