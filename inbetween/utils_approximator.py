import os
import json_tricks
from pathlib import Path


def load_targets():
    """
    Loads the target functions for approximation. Values taken from the
    posterior of a wide-limit BNN GP trained on a 1D regression dataset
    containing two separated clusters.
    Returns:
        X ([100, 1] numpy array): input x values on a 1D grid.
        mean ([100, 1] numpy array): target predictive mean
        var ([100, 1] numpy array): target predictive variance
    """
    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(os.path.dirname(filepath))  # Should be root
    approximator_path = Path(dirpath, "datasets", "approximator",
                             "approximator.json")
    with open(approximator_path, 'r') as file:
        data = json_tricks.load(file)
    return data["x_range"][:, None], data["mean"][:, None], data["var"][:, None]
