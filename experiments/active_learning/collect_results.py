import re
import argparse
from pathlib import Path
import os

if __name__ == '__main__':
    """Collect all the final rmses for one kind of inference for the full active
    learning experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference',
                        '-i',
                        type=str,
                        choices=['FFGBNN', 'DropoutBNN', 'GPBNN'],
                        help='Choice of inference. Default:'
                             ' ApproximatorFFGBNN')
    parser.add_argument('--acquisition',
                        '-a',
                        type=str,
                        choices = ['max_variance', 'random'],
                        help='Aquisition function used.')
    parser.add_argument('--active_dir',
                        type=str,
                        help='Path to directory containing the active learning '
                        'experiment results.')
    args = parser.parse_args()

    inference_dir = Path(args.active_dir, 'Naval', args.inference)
    mean_list, se_list = [], []
    for i in [1, 2, 3, 4]:
        summary_file = Path(inference_dir, f'{i}HL', args.acquisition,
                            'summary.txt')
        f = open(summary_file, "r")
        summary = f.read()
        # get final test rmse
        rmse_mean = (summary.split('RMSE: '))[1].split(' +-')[0]
        rmse_std = (summary.split('+- '))[1].split('\n')[0]
        rmse_mean = "{:.2f}".format(float(rmse_mean))
        rmse_std = "{:.2f}".format(float(rmse_std))
        mean_list.append(rmse_mean)
        se_list.append(rmse_std)
    output = ''
    for i, rmse in enumerate(mean_list):
        rmse_std = se_list[i]
        output = output + f' & ${rmse} \pm {rmse_std}$'
    print(output)
