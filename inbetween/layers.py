import torch
import numpy as np
from torch.nn import Module, Parameter
from torch.nn.functional import dropout, dropout2d
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from .utils import device


class Layer(Module):

    def __init__(self, input_dim, output_dim, sigma_w, sigma_b):
        """

        Args:
            input_dim: int, dimension of data that will be fed into the network
            output_dim: int, dimension of output of the network
            sigma_w: float, for each layer prior std for weights is
                     sigma_w / sqrt(layer.num_input)
            sigma_b: float, for each layer prior std for biases is sigma_b
        """

        super(Layer, self).__init__()
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.input_dim, self.output_dim = input_dim, output_dim
        self._build_prior(sigma_w, sigma_b)
        self._build_variational_posterior()

    def forward(self, input, num_samples):
        """
        Returns samples from variational posterior
        Args:
            input: [N, input_dim] or [num_sample, N, input_dim]
            num_samples: int, Number of samples from approx posterior to return
        Returns:
            Samples [num_samples, output_dim, N]
        """
        assert (len(input.shape) == 2 or input.shape[0] == num_samples)
        w_post, b_post = self.variational_posterior()
        w_samples = w_post.rsample((num_samples,))
        b_samples = b_post.rsample((num_samples,))
        return torch.matmul(input, w_samples) + b_samples

    def kl(self):
        """
        Returns: KL(Q || P), where P is the prior over the parameters and Q is
        the variational posterior. It is assumed that weights and biases are
        independent from each other under both P and Q
        """
        w_post, b_post = self.variational_posterior()
        w_kl = kl_divergence(w_post, self.w_prior).sum()
        b_kl = kl_divergence(b_post, self.b_prior).sum()
        return w_kl + b_kl

    def _build_prior(self, sigma_w, sigma_b):
        """
        Prior is constructed with Neal scaling for the weights
        Args:
            sigma_w: float
            sigma_b: float
        Returns: w_prior, b_prior, torch.distributions.Normal with
        0 mean and std sigma_w / sqrt(input_dim) and std sigma_b respectively
        """
        w_std = torch.ones(self.input_dim, self.output_dim)
        w_std = w_std * sigma_w / self.input_dim ** .5
        b_std = sigma_b * torch.ones(1, self.output_dim)
        w_prior = Normal(0., w_std)
        w_prior.loc = w_prior.loc.to(device)
        w_prior.scale = w_prior.scale.to(device)
        b_prior = Normal(0., b_std)
        b_prior.loc = b_prior.loc.to(device)
        b_prior.scale = b_prior.scale.to(device)
        self.w_prior = w_prior
        self.b_prior = b_prior

    def sample_prior(self, input, num_samples):
        """
        Returns samples from the prior
        Args:
            input: [input_dim, N] or [num_sample, input_dim, N]
            num_samples: int, Number of samples from approx posterior to return
        Returns:
            Samples [num_samples, N, output_dim]
        """
        assert (len(input.shape) == 2 or input.shape[0] == num_samples)

        self._build_prior(self.sigma_w, self.sigma_b)
        w_samples = self.w_prior.rsample((num_samples,))
        b_samples = self.b_prior.rsample((num_samples,))
        return torch.matmul(input, w_samples) + b_samples

    def _build_variational_posterior(self):
        raise NotImplementedError

    def variational_posterior(self):
        raise NotImplementedError


class FFGLayer(Layer):

    def __init__(self, input_dim, output_dim, sigma_w, sigma_b):
        super(FFGLayer, self).__init__(input_dim, output_dim, sigma_w, sigma_b)

    def _set_prior(self):
        """Set the variational posterior to be the prior"""
        w_mean = self.w_prior.loc.detach().clone()
        w_logstd = torch.log(self.w_prior.scale.detach().clone())
        b_mean = self.b_prior.loc.detach().clone()
        b_logstd = torch.log(self.b_prior.scale.detach().clone())
        self.w_mean = Parameter(w_mean)
        self.w_logstd = Parameter(w_logstd)
        self.b_mean = Parameter(b_mean)
        self.b_logstd = Parameter(b_logstd)

    def _build_variational_posterior(self):
        """ Initialise following Tomczak et al 2018 'Neural network ensembles
        and variational inference revisited'
        """
        w_mean_scale = 1 / np.sqrt(2 * self.output_dim)
        w_mean = torch.normal(0., w_mean_scale,
                              (self.input_dim, self.output_dim))
        b_mean = torch.zeros(1, self.output_dim)
        logstd = -5 * np.log(10.)
        w_logstd = logstd * torch.ones(self.input_dim, self.output_dim)
        b_logstd = logstd * torch.ones(1, self.output_dim)
        self.w_mean = Parameter(w_mean.to(device))
        self.w_logstd = Parameter(w_logstd.to(device))
        self.b_mean = Parameter(b_mean.to(device))
        self.b_logstd = Parameter(b_logstd.to(device))

    def variational_posterior(self):
        """
        Returns: torch.distribution.normal [output_dim, input_dim],
                 torch.distribution.normal [output_dim]
        """
        w_post = Normal(self.w_mean, torch.exp(self.w_logstd))
        b_post = Normal(self.b_mean, torch.exp(self.b_logstd))
        return w_post, b_post

    def forward(self, x_data, num_samples, local=True):
        """ Returns samples from variational posterior. If local is true, samples
        are only valid marginally
        Args:
            x_data: [input_dim, N] or [num_sample, output_dim, N]
            num_samples: int, number of samples
            local: boolean, if True "local reparameterization" is used.

        Returns: Samples [num_samples, output_dim, N]

        """
        assert (len(x_data.shape) == 2 or x_data.shape[0] == num_samples)
        return self._local_forward(x_data, num_samples) if local \
            else super(FFGLayer, self).forward(x_data, num_samples)

    def _local_forward(self, x_data, num_samples):
        """
        Samples variational posterior at x_data using local reparam
        Note: Samples are not coherent at different input points when local
        reparam is used

        Args:
            x_data: [N, input_dim] or [num_sample, output_dim, N]
            num_samples: int, number of samples

        Returns: Samples [num_samples, output_dim, N]
        """
        assert x_data.shape[-1] == self.input_dim
        assert x_data.shape[0] == num_samples or len(x_data.shape) == 2
        # local reparameterisation trick
        b_std = torch.exp(self.b_logstd)
        w_std = torch.exp(self.w_logstd)
        act_mean = torch.matmul(x_data, self.w_mean) + self.b_mean
        act_var = torch.matmul(x_data ** 2, w_std ** 2) + b_std ** 2
        act_std = torch.sqrt(act_var)
        act_dist = Normal(act_mean, act_std)
        return act_dist.rsample() if len(
            act_mean.shape) == 3 else act_dist.rsample([num_samples])


class DropoutLayer(Layer):

    def __init__(self, input_dim, output_dim, sigma_w, sigma_b, dropout_rate):
        super(DropoutLayer, self).__init__(input_dim, output_dim, sigma_w,
                                           sigma_b)
        self.dropout_rate = dropout_rate
        self.w_l2, self.b_l2 = self._calculate_reg()

    def _build_variational_posterior(self):
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        self.linear = self.linear.to(device)

    def _calculate_reg(self):
        """
        Returns: w_l2, b_l2, float, float. Weight and bias decay parameters,
        given in Gal, 2016 http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf
        section 3.2.3

        """
        w_prior_var = self.sigma_w ** 2 / self.input_dim
        b_prior_var = self.sigma_b ** 2
        w_l2 = (1. - self.dropout_rate) / (2 * w_prior_var)
        b_l2 = 1 / (2 * b_prior_var)
        return w_l2, b_l2

    def kl(self):
        """ KL divergence up to an 'infinite constant', to match 'KL condition'
        in Gal 2016 PhD thesis
        """
        w_kl = self.w_l2 * torch.sum(self.linear.weight ** 2)
        b_kl = self.b_l2 * torch.sum(self.linear.bias ** 2)
        return w_kl + b_kl

    def forward(self, x_data, num_samples, local=False):
        if self.dropout_rate == 0:
            return self.linear(x_data)
        if len(x_data.shape) == 2:
            x_data = x_data[None, ...].expand(num_samples, -1, -1)
        if local:
            dropped_inputs = dropout(x_data, self.dropout_rate, True, False)
        else:
            x_data = torch.transpose(x_data, 1, 2)[..., None]
            dropped_inputs = dropout2d(x_data, self.dropout_rate, True, False)[
                ..., 0]
            dropped_inputs = torch.transpose(dropped_inputs, 1, 2)
        dropped_inputs *= (1. - self.dropout_rate)
        return self.linear(dropped_inputs)
