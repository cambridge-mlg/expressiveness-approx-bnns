import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from active_learners import ActiveLearner

from inbetween.utils import load_bb


class ActiveExperiment:

    def __init__(self, dataset, no_splits, no_init_points, logdir, no_iters):
        self.no_splits = no_splits
        self.no_init_points = no_init_points
        self.no_iters = no_iters
        self.logdir = logdir
        self.dataset = dataset
        self.test_prop = .1
        self.test_metrics = {"rmses": dict(), "lls": dict()}
        self.init_indices, self.test_indices = dict(), dict()

    def _load_data(self, split):
        return load_bb(self.dataset, split, 1. - self.test_prop)

    def _init_indices_for_split(self, split, num_points):
        np.random.seed(split)
        return np.random.choice(num_points, self.no_init_points, replace=False)

    def run(self, model_parameters, training_parameters, active_batch_size,
            acquisition_function):
        for i in range(self.no_splits):
            start = time.time()
            filepath = Path(self.logdir, f"split{i}")
            os.makedirs(filepath, exist_ok=True)
            (X_train, y_train), (X_test, y_test) = self._load_data(i)
            init_indices = self._init_indices_for_split(i, X_train.shape[0])
            active_learner = ActiveLearner(X_train=X_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test,
                                           init_indices=init_indices,
                                           model_parameters=model_parameters,
                                           training_parameters=training_parameters,
                                           no_iters=self.no_iters,
                                           active_batch_size=active_batch_size,
                                           logdir=filepath,
                                           acquisition_function=acquisition_function)

            active_learner.learn(seed=i)
            for k, v in active_learner.test_metrics.items():
                self.test_metrics[k][i] = v
            print(f'TESTS ON SPLIT {i} COMPLETE')
            print(f'TIME FOR SPLIT: {time.time() - start}')

    def plot_all_results(self):
        # calculate +/- one standard error bars and plot
        all_lls = np.array(
            [self.test_metrics["lls"][i] for i in range(self.no_splits)])
        all_rmses = np.array(
            [self.test_metrics["rmses"][i] for i in range(self.no_splits)])
        mean_test_lls = np.mean(all_lls, axis=0)
        mean_test_rmses = np.mean(all_rmses, axis=0)

        se_test_lls = np.std(all_lls, axis=0) / np.sqrt(self.no_splits)
        se_test_rmses = np.std(all_rmses, axis=0) / np.sqrt(self.no_splits)

        # plus one due to final prediction at the end of active learning
        plt.plot(np.arange(self.no_iters + 1), mean_test_lls)
        plt.fill_between(np.arange(self.no_iters + 1),
                         mean_test_lls + se_test_lls,
                         mean_test_lls - se_test_lls, alpha=0.3)
        plt.xlabel('Iterations of active learning')
        plt.ylabel('Average test log-likelihood')
        plt.savefig(Path(self.logdir, 'test_lls.png'))
        plt.close()

        # plus one due to final prediction at the end of active learning
        plt.plot(np.arange(self.no_iters + 1), mean_test_rmses)
        plt.fill_between(np.arange(self.no_iters + 1),
                         mean_test_rmses + se_test_rmses,
                         mean_test_rmses - se_test_rmses,
                         alpha=0.3)
        plt.xlabel('Iterations of active learning')
        plt.ylabel('Average test RMSE')
        plt.savefig(Path(self.logdir, 'test_rmses.png'))
        plt.close()
        # for plotting later on
        with open(Path(self.logdir, 'test_results.pkl'), "wb") as f:
            pickle.dump(self.test_metrics, f)
        # for easy inspection of final values
        final_rmse = mean_test_rmses[-1]
        final_rmse_se = se_test_rmses[-1]
        rmse_str = f"Final test RMSE: {final_rmse} +- {final_rmse_se}\n"
        final_ll = mean_test_lls[-1]
        final_ll_se = se_test_lls[-1]
        ll_str = f"Final test LL: {final_ll} +- {final_ll_se}\n"
        with open(Path(self.logdir, "summary.txt"), "a") as text_file:
            text_file.write(rmse_str)
            text_file.write(ll_str)
