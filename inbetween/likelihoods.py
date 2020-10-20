import torch.distributions
from torch.nn import Module


class Likelihood(Module):
    def __init__(self):
        super(Likelihood, self).__init__()

    def forward(self, outputs, y_data):
        raise NotImplementedError


class HomoskedasticGaussianRegression(Likelihood):

    def __init__(self, noise_std):
        self.noise_std = noise_std
        self.likelihood_dist = torch.distributions.Normal(loc=0., scale=noise_std)
        super(HomoskedasticGaussianRegression, self).__init__()

    def forward(self, outputs, y_data):
        errors = y_data - outputs
        return self.likelihood_dist.log_prob(errors)
