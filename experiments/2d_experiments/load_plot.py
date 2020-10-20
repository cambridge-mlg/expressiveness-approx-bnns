import argparse
from pathlib import Path
from inbetween.utils_2d import load_2d_pred, make_2d_plot

if __name__ == '__main__':
    """Load data and generate 2D plot"""
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
                        choices=['GPBNN', 'FFGBNN', 'DropoutBNN', 'HMCBNN'],
                        help='Inference method being loaded.')
    parser.add_argument('--datapath',
                        type=str,
                        help='Path to pickle file containing data for 2D'
                             ' plotting.')
    parser.add_argument('--figpath',
                        type=str,
                        help='Path to directory where figure is saved.')
    args = parser.parse_args()

    pickle_path = Path(args.datapath)
    hmc = (args.inference == 'HMCBNN')
    contour_std, slice_mean, slice_std = load_2d_pred(pickle_path, hmc=hmc)
    figpath = Path(args.figpath, f'{args.dataset}_{args.inference}_2d.pdf')
    make_2d_plot(contour_std, slice_mean, slice_std, args.dataset, figpath)
