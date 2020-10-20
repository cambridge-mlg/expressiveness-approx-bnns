import warnings
warnings.simplefilter("ignore")

from inbetween.models import FFGBNN, GPBNN
from inbetween.likelihoods import HomoskedasticGaussianRegression
from torch.nn.functional import relu
import tensorflow as tf
from tensorflow_probability import distributions
from inbetween.utils import load_bb
import pytest


def test_log_density():
    """ Compare the log-density method obtained by sampling wide BNN versus
    an analytic GP.
    """
    (X_train, y_train), (X_test, y_test) = load_bb('Boston', 0, 0.2)
    noise_std = .1
    depth = 2
    sigma_w = 1.
    sigma_b = 1.
    gp = GPBNN(input_dim=X_train.shape[-1],
               likelihood=HomoskedasticGaussianRegression(noise_std),
               nonlinearity=relu,
               num_layers=depth,
               num_train=X_train.shape[0],
               output_dim=1,
               sigma_w=sigma_w,
               sigma_b=sigma_b)
    vars = gp.kernel(X_train, full_cov=False) + noise_std ** 2
    scale = tf.math.sqrt(vars)
    loc = tf.zeros(X_train.shape[0], dtype=tf.dtypes.float64)
    prior = distributions.Normal(loc, scale)
    GP_LPD = prior.log_prob(y_train[:, 0]).numpy()

    bnn = FFGBNN(input_dim=X_train.shape[-1],
                 likelihood=HomoskedasticGaussianRegression(noise_std),
                 num_train=0,
                 output_dim=1,
                 nonlinearity=relu,
                 num_layers=depth,
                 width=500,
                 sigma_w=sigma_w,
                 sigma_b=sigma_b)
    bnn.set_prior()
    BNN_LPD = bnn.predict_log_density(X_train, y_train, num_samples=2000).squeeze()
    assert BNN_LPD.mean() == pytest.approx(GP_LPD.mean(), 0.1)
