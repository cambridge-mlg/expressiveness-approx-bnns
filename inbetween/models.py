import numpy as np
import math
import torch
import torch.optim as optim
from torch.nn.functional import relu
from torch.nn import Module, ModuleList
from .layers import FFGLayer, DropoutLayer
from inbetween.utils import device, process_data, process_data_approximator
from inbetween.NNKernel import ReLUKernel
from inbetween.likelihoods import HomoskedasticGaussianRegression
from inbetween.losses import elbo_batch_loss, approximator_batch_loss
import gpflow
import time


class BNN(Module):

    def __init__(self, input_dim, output_dim, likelihood, nonlinearity,
                 num_train):
        super(BNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.likelihood = likelihood
        self.nonlinearity = nonlinearity
        self.num_train = num_train
        self.batch_loss = elbo_batch_loss
        self.process_data = process_data

    def forward(self, x_data, num_samples, **kwargs):
        """Forward pass.
        Args:
            x_data: [N, input_dim] numpy/torch array
            num_samples: int
        Returns:
            output: [N, output_dim] numpy/torch array
        """
        assert len(x_data.shape) == 2
        assert x_data.shape[-1] == self.input_dim
        torch_input = torch.is_tensor(x_data)
        if not torch_input:
            x_data = torch.Tensor(x_data).to(device)

        for i, layer in enumerate(self.layers):
            x_data = layer(x_data, num_samples, **kwargs)
            if i < len(self.layers) - 1:
                x_data = self.nonlinearity(x_data)
        return x_data if torch_input else x_data.detach().cpu().numpy()

    def predict_log_density(self, Xnew, ynew, batch_size=None, local=True,
                            **kwargs):
        """
        Point-wise log density estimate at ynew under dist. of f(Xnew) + noise
        Args:
            Xnew: [N, D_in], numpy.array or torch.Tensor
            ynew: [N, D_out], numpy.array or torch.Tensor
            batch_size: int, minibatch size, only used if local=True
            local: bool, whether to use local reparameterisation. If true,
            log density is computed via minibatching over data points

        Returns: [N] log density

        """
        torch_input = torch.is_tensor(Xnew) and torch.is_tensor(ynew)
        if not torch_input:
            Xnew = torch.Tensor(Xnew).to(device)
            ynew = torch.Tensor(ynew).to(device)
        if local and batch_size is not None:
            # only minibatches if using local reparam
            num_points = Xnew.shape[0]
            num_batch = math.ceil(num_points / batch_size)
            ll_batch = []
            for i in range(num_batch):
                start = i * batch_size
                end = start + batch_size
                f_samples = self.forward(Xnew[start:end, :],
                                         local=local,
                                         **kwargs).detach()
                log_density_samples = self.likelihood(f_samples,
                                                      ynew[start:end, :])
                log_density = torch.logsumexp(log_density_samples,
                                              dim=0,
                                              keepdim=False)
                log_density -= np.log(float(log_density_samples.size()[0]))
                ll_batch.append(log_density)
            log_density = torch.cat(ll_batch, dim=0)
        elif not local and batch_size is not None:
            raise Exception('Code does not minibatch if not using local '
                            'reparam, since the function samples will '
                            'lose coherence')
        else:
            f_samples = self.forward(Xnew, local=local, **kwargs)
            log_density_samples = self.likelihood(f_samples, ynew)
            log_density = torch.logsumexp(log_density_samples,
                                          dim=0,
                                          keepdim=False)
            log_density -= np.log(float(log_density_samples.size()[0]))
        if torch_input:
            return log_density
        else:
            return log_density.detach().cpu().numpy()

    def pred_mean_std(self, Xnew, add_likelihood_noise=False, batch_size=None,
                      local=True, **kwargs):
        """
        Estimate posterior predictive mean and variance with samples
        Args:
            Xnew: [N, D], numpy.array or torch.Tensor
            add_likelihood_noise: bool, if True, estimate mean/variance of y,
            if false, estimate mean and variance of f.
            batch_size: int, minibatch size, only used if local=True
            local: bool, whether to use local reparameterisation. If true,
            log density is computed via minibatching over data points

        Returns: [N], [N] MC estimates of Marginal mean and variance at Xnew

        """
        torch_input = torch.is_tensor(Xnew)
        if not torch_input:
            Xnew = torch.Tensor(Xnew).to(device)
        # only minibatch if doing local reparam, since we don't need coherent
        # samples in that case.
        if local and batch_size is not None:
            num_points = Xnew.shape[0]
            num_batch = math.ceil(num_points / batch_size)
            mean_batch, std_batch = [], []
            for i in range(num_batch):
                start = i * batch_size
                end = start + batch_size
                f_out = self.forward(Xnew[start:end, :],
                                     local=local,
                                     **kwargs).detach()
                mean_batch.append(f_out.mean(0))
                std_batch.append(f_out.std(0))
            mean = torch.cat(mean_batch, dim=0)
            std = torch.cat(std_batch, dim=0)
        elif not local and batch_size is not None:
            raise Exception('Code does not minibatch if not using local '
                            'reparam, since the function samples will '
                            'lose coherence')
        else:  # get all samples at once so that they're coherent
            all_out = self.forward(Xnew, local=local, **kwargs).detach()
            mean = all_out.mean(0)
            std = all_out.std(0)
        if add_likelihood_noise:
            if isinstance(self.likelihood, HomoskedasticGaussianRegression):
                std = torch.sqrt(std ** 2 + self.likelihood.noise_std ** 2)
            else:
                raise NotImplementedError
        if not torch_input:
            mean, std = mean.detach().cpu().numpy(), std.detach().cpu().numpy()
        return mean, std

    def train(self, X, y, num_epochs, batch_size, samples, lr, verbose=False):
        """Train the model using Adam.
        Args:
            X: [N, input_dim] numpy array
            y: [N, output_dim] numpy array for standard training, OR 2 element
                list for direct approximation training, with first element
                target mean, and second element target variance.
            iters: int, number of optimisation steps
            batch_size: int
            samples: int
            lr: float, learning rate
            verbose: bool, if True, print loss and timer
        """
        X, y = self.process_data(self, X, y)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if verbose:
            start = time.time()
        for epoch in range(num_epochs):
            permutation = torch.randperm(X.size()[0])
            for i in np.arange(0, X.size()[0], batch_size):
                optimizer.zero_grad()
                inds = permutation[i:i + batch_size]
                loss = self.batch_loss(self, X, y, inds, samples)
                loss.backward()
                optimizer.step()
                if verbose and epoch % 500 == 0:
                    loss_np = loss.detach().cpu().numpy()
                    print(f'Epoch:{epoch}, Loss:{loss_np}')
        if verbose:
            print(f'Train time: {time.time() - start}')

    def _kl(self):
        return torch.sum(torch.stack([layer.kl() for layer in self.layers]))


class FFGBNN(BNN):

    def __init__(self, input_dim, output_dim, likelihood, nonlinearity,
                 num_layers, width, num_train, sigma_w, sigma_b):
        super(FFGBNN, self).__init__(input_dim, output_dim, likelihood,
                                     nonlinearity, num_train=num_train)
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.width = width
        self.num_layers = num_layers
        self.layers = self._initialize_layers(num_layers, width, sigma_w,
                                              sigma_b)

    def sample_prior(self, x_data, num_samples):
        """Sample from the BNN prior.
        Args:
            x_data: [N, input_dim] numpy/torch array
            num_samples: int
        Returns:
            output: [N, output_dim] numpy/torch array
        """
        assert len(x_data.shape) == 2
        assert x_data.shape[-1] == self.input_dim
        torch_input = torch.is_tensor(x_data)
        if not torch_input:
            x_data = torch.Tensor(x_data).to(device)

        for i, layer in enumerate(self.layers):
            x_data = layer.sample_prior(x_data, num_samples)
            if i < len(self.layers) - 1:
                x_data = self.nonlinearity(x_data)
        return x_data if torch_input else x_data.detach().cpu().numpy()

    def set_prior(self):
        """Set the MFVI variational posterior to the prior"""
        for layer in self.layers:
            layer._set_prior()

    def predict_log_density(self, Xnew, ynew, num_samples=200, batch_size=None,
                            local=False):
        return super(FFGBNN, self).predict_log_density(Xnew, ynew,
                                                       num_samples=num_samples,
                                                       batch_size=batch_size,
                                                       local=local)

    def pred_mean_std(self, Xnew, add_likelihood_noise=False, batch_size=None,
                      num_samples=200, local=False):
        return super(FFGBNN, self).pred_mean_std(Xnew, num_samples=num_samples,
                                                 batch_size=batch_size,
                                                 local=local)

    def _initialize_layers(self, num_layers, width, sigma_w, sigma_b):
        layers = ModuleList([])
        for i in range(num_layers + 1):
            input_dim = self.input_dim if i == 0 else width
            output_dim = self.output_dim if i == num_layers else width
            layer = FFGLayer(input_dim, output_dim, sigma_w, sigma_b)
            layers.append(layer)
        return layers

    def reset(self):
        self.layers = self._initialize_layers(self.num_layers, self.width,
                                              self.sigma_w, self.sigma_b)


class DropoutBNN(BNN):

    def __init__(self, input_dim, output_dim, likelihood, nonlinearity,
                 num_layers, width, num_train, sigma_w, sigma_b,
                 dropout_rate=.05, dropout_bottom=False):
        super(DropoutBNN, self).__init__(input_dim, output_dim, likelihood,
                                         nonlinearity, num_train)
        self.layers = self._initialize_layers(num_layers, width, sigma_w,
                                              sigma_b, dropout_rate,
                                              dropout_bottom)
        self.num_layers = num_layers
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.width = width
        self.dropout_rate = dropout_rate
        self.dropout_bottom = dropout_bottom

    def _initialize_layers(self, num_layers, width, sigma_w, sigma_b,
                           dropout_rate, dropout_bottom):
        layers = ModuleList([])
        for i in range(num_layers + 1):
            input_dim = self.input_dim if i == 0 else width
            output_dim = self.output_dim if i == num_layers else width
            if i == 0 and not dropout_bottom:
                layer = DropoutLayer(input_dim, output_dim, sigma_w, sigma_b,
                                     0.)
            else:
                layer = DropoutLayer(input_dim, output_dim, sigma_w, sigma_b,
                                     dropout_rate)
            layers.append(layer)
        return layers

    def pred_mean_std(self, Xnew, add_likelihood_noise=False, batch_size=None,
                      num_samples=200, local=False):
        return super(DropoutBNN, self).pred_mean_std(Xnew,
                                                     num_samples=num_samples,
                                                     batch_size=batch_size,
                                                     local=local)

    def predict_log_density(self, Xnew, ynew, num_samples=200, batch_size=None,
                            local=False):
        return super(DropoutBNN, self).predict_log_density(Xnew, ynew,
                                                           num_samples=num_samples,
                                                           batch_size=batch_size,
                                                           local=local)

    def reset(self):
        self.layers = self._initialize_layers(self.num_layers, self.width,
                                              self.sigma_w, self.sigma_b,
                                              self.dropout_rate,
                                              self.dropout_bottom)


class GPBNN(BNN):

    def __init__(self, input_dim, output_dim, likelihood, nonlinearity,
                 num_layers, num_train, sigma_w, sigma_b):
        super(GPBNN, self).__init__(input_dim, output_dim, likelihood,
                                    nonlinearity, num_train)
        assert isinstance(likelihood, HomoskedasticGaussianRegression)
        self.noise_std = likelihood.likelihood_dist.scale
        self.input_dim = input_dim
        if nonlinearity == relu:
            self.kernel = ReLUKernel(prior_weight_std=sigma_w,
                                     prior_bias_std=sigma_b,
                                     depth=num_layers)
        else:
            raise NotImplementedError
        self.gp = gpflow.models.GPR(data=(np.ones((0, input_dim)),
                                          np.ones((0, input_dim))),
                                    kernel=self.kernel,
                                    noise_variance=self.noise_std ** 2)

    def forward(self, x_data, num_samples, **kwargs):
        assert not torch.is_tensor(x_data)
        return self.pred_sample(x_data, num_samples)

    def train(self, X, y, **kwargs):
        assert not torch.is_tensor(X) and not torch.is_tensor(y)
        self.gp = gpflow.models.GPR((X, y), kernel=self.kernel,
                                    noise_variance=self.noise_std ** 2.)

    def reset(self):
        self.gp = gpflow.models.GPR(data=(np.ones((0, self.input_dim)),
                                          np.ones((0, self.input_dim))),
                                    kernel=self.kernel,
                                    noise_variance=self.noise_std ** 2)

    def pred_mean_std(self, X_test, add_likelihood_noise=False, **kwargs):
        assert not torch.is_tensor(X_test)
        mean, var = self.gp.predict_y(X_test) if add_likelihood_noise \
            else self.gp.predict_f(X_test)
        return mean.numpy(), np.sqrt(var)

    def pred_sample(self, X_test, num_samples):
        assert not torch.is_tensor(X_test)
        return self.gp.predict_f_samples(X_test, num_samples).numpy()

    def predict_log_density(self, Xnew, ynew, **kwargs):
        assert not torch.is_tensor(Xnew)
        assert not torch.is_tensor(ynew)
        return self.gp.predict_log_density((Xnew, ynew))


class ApproximatorFFGBNN(FFGBNN):

    def __init__(self, input_dim, output_dim, nonlinearity, num_layers, width,
                 num_train):
        super(ApproximatorFFGBNN, self).__init__(input_dim=input_dim,
                                                 output_dim=output_dim,
                                                 likelihood=None,
                                                 nonlinearity=nonlinearity,
                                                 num_layers=num_layers,
                                                 width=width,
                                                 num_train=num_train,
                                                 sigma_w=1., sigma_b=1.)
        self.process_data = process_data_approximator
        self.batch_loss = approximator_batch_loss
        self.set_prior()


class ApproximatorDropoutBNN(DropoutBNN):

    def __init__(self, input_dim, output_dim, nonlinearity, num_layers, width,
                 num_train, dropout_rate=.05, dropout_bottom=False):
        super(ApproximatorDropoutBNN, self).__init__(input_dim=input_dim,
                                                     output_dim=output_dim,
                                                     likelihood=None,
                                                     nonlinearity=nonlinearity,
                                                     num_layers=num_layers,
                                                     width=width,
                                                     num_train=num_train,
                                                     sigma_w=1., sigma_b=1.,
                                                     dropout_rate=dropout_rate,
                                                     dropout_bottom=dropout_bottom)
        self.process_data = process_data_approximator
        self.batch_loss = approximator_batch_loss
