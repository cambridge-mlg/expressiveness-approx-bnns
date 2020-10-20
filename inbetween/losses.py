import torch
# loss functions


def elbo(model, output, target):
    minibatch_size = target.shape[0]
    weight = model.num_train / minibatch_size
    ll = model.likelihood(output, target).sum()
    return weight * ll / output.shape[0] - model._kl()


def elbo_batch_loss(model, X, y, inds, samples):
    """Compute an estimate of the negative ELBO loss from a single batch.
    Args:
        X ([N, input_dim] numpy array): Train inputs.
        y ([N, output_dim] numpy array): Train outputs.
        inds ([batch_size] torch tensor): Indices for the batch.
        samples (int): Number of MC samples to estimate expected
            log-likelihood term of the ELBO.
    Returns: -elbo ([1] torch tensor): Minibatch estimate of the negative
        ELBO.
    """
    X_batch, y_batch = X[inds], y[inds]
    output = model.forward(X_batch, samples, local=True)
    return -elbo(model, output, y_batch)


def approximator_batch_loss(model, X, target, inds, samples):
    """Loss for directly minimising squared error between BNN predictive
    and target mean and variance function.
    Args:
        X ([N, input_dim] numpy array): Train inputs.
        target (list): First element is ([N, output_dim]) target mean,
            second element is ([N, output_dim]) target variance.
        inds ([batch_size] torch tensor): Indices for the batch.
        samples (int): Number of MC samples to estimate expected
            log-likelihood term of the ELBO.
    Returns: ([1] torch tensor): Minibatch estimate of the sum of
    squared error loss.

    NOTE: this is not strictly an unbiased estimator due to the square root
    in the variance loss. Best to perform full-batch training.
    """
    assert len(target) == 2  # a target mean and a target variance
    target_mean = target[0][inds]
    target_var = target[1][inds]
    batch_X = X[inds]
    output = model.forward(batch_X, samples, local=True)
    output_mean = output.mean(0)
    output_var = output.var(0)

    minibatch_size = batch_X.shape[0]
    weight = model.num_train / minibatch_size
    mean_loss = torch.sum((output_mean - target_mean) ** 2)
    var_loss = torch.sum((output_var - target_var) ** 2)
    return weight * (mean_loss + var_loss)
