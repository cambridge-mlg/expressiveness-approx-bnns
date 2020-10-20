import torch
import bayesian_benchmarks.data as bbd
from torch.nn.functional import relu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model_parameters(args):
    model_params, training_params = dict(), dict()

    model_params['inference_type'] = args.inference
    model_params["num_layers"] = args.layers
    model_params["nonlinearity"] = relu

    if args.inference not in ["ApproximatorDropoutBNN", "ApproximatorFFGBNN"]:
        # if not an approximator
        model_params["noise_std"] = args.noise_std
        model_params["sigma_w"] = args.sigma_w
        model_params["sigma_b"] = args.sigma_b

    if args.inference != "GPBNN":
        model_params['width'] = args.width
        training_params['lr'] = args.learning_rate
        training_params['num_epochs'] = args.num_epochs
        training_params['samples'] = args.samples
        training_params['batch_size'] = args.minibatch_size

    if args.inference in ["DropoutBNN", "ApproximatorDropoutBNN"]:
        # if dropout used
        model_params["dropout_rate"] = args.dropout_rate
        model_params["dropout_bottom"] = args.dropout_bottom

    return model_params, training_params


def load_bb(dataset, split, prop):
    d = getattr(bbd, dataset)(split, prop)
    return (d.X_train, d.Y_train), (d.X_test, d.Y_test)


def process_data(model, X, y):
    """Assert training y has the right shape and convert to torch tensor for
    standard BNN.
    Args:
        model (BNN object): The BNN.
        y ([N, output_dim] numpy array): Training set outputs.
    Returns:
        y ([N, output_dim] torch tensor): Training set outputs.
    """
    assert len(X.shape) == 2
    assert X.shape[-1] == model.input_dim
    X = torch.Tensor(X).to(device)
    assert len(y.shape) == 2
    assert y.shape[-1] == model.output_dim
    y = torch.Tensor(y).to(device)
    return X, y


def process_data_approximator(model, X, targets):
    """Assert training targets has is a list and convert to torch tensor for
    approximator BNN that directly matches a target mean and variance.
    Args:
        model (ApproximatorBNN object): The approximator BNN.
        targets (list of numpy arrays): 2 element list for direct approximation
            training, with first element target mean, and second element target
            variance.
    Returns:
        targets (list of torch tensors): Converted to tensors.
    """
    assert len(X.shape) == 2
    assert X.shape[-1] == model.input_dim
    X = torch.Tensor(X).to(device)
    assert len(targets) == 2
    targets = [torch.Tensor(array).to(device) for array in targets]
    assert targets[0].shape[-1] == model.output_dim
    assert targets[1].shape[-1] == model.output_dim
    return X, targets
