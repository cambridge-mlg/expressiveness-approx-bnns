import os
import pickle
from pathlib import Path
import numpy as np
import torch

from inbetween.likelihoods import HomoskedasticGaussianRegression
import inbetween.models


class ActiveLearner:
    def __init__(self, X_train, y_train, X_test, y_test,
                 init_indices, model_parameters, training_parameters, no_iters,
                 active_batch_size, logdir, acquisition_function):
        # Directory to save images
        self.logdir = logdir
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.init_indices = init_indices

        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.no_iters, self.active_batch_size = no_iters, active_batch_size
        self.acquisition_function = acquisition_function
        self.test_metrics = dict(lls=list(), rmses=list())
        self.noise_std = self.model_parameters.pop("noise_std")
        self.likelihood = HomoskedasticGaussianRegression(noise_std=
                                                          self.noise_std)
        self.model = self._build_model()
        self.model_parameters["noise_std"] = self.noise_std

        # for plotting later on
        os.makedirs(logdir, exist_ok=True)
        data = dict(train_data=(X_train, y_train), test_data=(X_test, y_test))
        with open(Path(logdir, 'data.pkl'), "wb") as f:
            pickle.dump(data, f)

    def _build_model(self):
        inference = self.model_parameters.pop('inference_type')
        model = getattr(inbetween.models, inference)(likelihood=self.likelihood,
                                                     input_dim=
                                                     self.X_train.shape[-1],
                                                     num_train=0,
                                                     output_dim=
                                                     self.y_train.shape[-1],
                                                     **self.model_parameters)
        self.model_parameters["inference_type"] = inference
        return model

    def learn(self, seed=0):
        # set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        # initialise indices
        active_indices = self.init_indices
        # points under consideration to add to the train set
        all_indices = np.arange(len(self.X_train))
        inactive_indices = np.setdiff1d(all_indices, active_indices)
        for i in range(self.no_iters):
            prediction_outputs = self._train_and_predict(active_indices)
            # Select points to be added to train set
            uncertainty = prediction_outputs["f_std_train"]
            new_active_indices = self._acquire_points(inactive_indices,
                                                      uncertainty)
            # Remove new active indices from inactive indices
            inactive_indices = np.setdiff1d(inactive_indices,
                                            new_active_indices)
            # Compute test performance on given indices and make plots
            self._save(i, prediction_outputs, active_indices,
                       new_active_indices, inactive_indices)
            # Add new points to the training set
            active_indices = np.concatenate(
                (active_indices, new_active_indices))
            # add test performance to lists
            for metric in self.test_metrics:
                self.test_metrics[metric].append(prediction_outputs[metric])
        # Perform one final round of training and prediction
        prediction_outputs = self._train_and_predict(active_indices)
        # add test performance to lists
        for metric in self.test_metrics:
            self.test_metrics[metric].append(prediction_outputs[metric])
        new_active_indices = []  # no new indices in final iter
        # Remove new active indices from inactive indices
        inactive_indices = np.setdiff1d(all_indices,
                                        active_indices)
        # Compute test performance on given indices and make plots
        self._save(self.no_iters, prediction_outputs, active_indices,
                   new_active_indices, inactive_indices)

    def _train_and_predict(self, active_indices):
        # train on the current active set and make predictions
        X_active = self.X_train[active_indices]
        y_active = self.y_train[active_indices]
        self.model.reset()
        self.model.num_train = len(X_active)
        self.model.train(X_active, y_active, **self.training_parameters)
        return self.predict()

    def _acquire_points(self, inactive_indices, uncertainty):
        # Select candidate points with highest std
        if self.acquisition_function == 'max_variance':
            uncertainty_inactive = uncertainty[inactive_indices]
            inds = np.argsort(uncertainty_inactive[:, 0])[-self.active_batch_size:]
        elif self.acquisition_function == 'random':
            inds = np.random.choice(len(inactive_indices),
                                    self.active_batch_size, replace=False)
        else:
            raise NotImplementedError
        return inactive_indices[inds]

    def _save(self, iter, prediction_outputs, active_indices,
              new_active_indices, inactive_indices):
        # Save absolute errors for possible future plotting
        train_errors = np.abs(prediction_outputs["f_mean_train"] - self.y_train)
        test_errors = np.abs(prediction_outputs["f_mean_test"] - self.y_test)
        data = dict(init_indices=self.init_indices,
                    active_indices=active_indices,
                    new_active_indices=new_active_indices,
                    inactive_indices=inactive_indices,
                    f_std_train=prediction_outputs["f_std_train"],
                    f_std_test=prediction_outputs["f_std_test"],
                    f_mean_train=prediction_outputs["f_mean_train"],
                    f_mean_test=prediction_outputs["f_mean_test"],
                    train_errors=train_errors,
                    test_errors=test_errors)
        os.makedirs(Path(self.logdir, "logged-data"), exist_ok=True)
        with open(Path(self.logdir, "logged-data", f"{iter}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def predict(self):
        predict_outputs = dict()
        # Predict on training data
        f_mean_train, f_std_train = self.model.pred_mean_std(self.X_train,
                                                             batch_size=64,
                                                             num_samples=500,
                                                             local=True)
        predict_outputs["f_mean_train"] = f_mean_train
        predict_outputs["f_std_train"] = f_std_train
        # Predict on test data, evaluate rmse and nlpd
        predict_outputs["lls"] = np.mean(
            self.model.predict_log_density(self.X_test, self.y_test,
                                           batch_size=64,
                                           num_samples=500,
                                           local=True))
        f_mean_test, f_std_test = self.model.pred_mean_std(self.X_test)
        predict_outputs["f_mean_test"] = f_mean_test
        predict_outputs["f_std_test"] = f_std_test
        square_errors = np.square(f_mean_test - self.y_test)
        predict_outputs["rmses"] = np.sqrt(np.mean(square_errors))
        return predict_outputs
