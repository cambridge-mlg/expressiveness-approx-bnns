import pytest
from pytest import approx
from inbetween.utils import load_bb
import numpy as np


def is_normalized(dataset):
    (X_train, Y_train), (X_test, Y_test) = load_bb(dataset, split=0, prop=.8)
    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ymu = approx(np.mean(Y), abs=1e-4) == 0.
    ystd = approx(np.std(Y), rel=1e-4) == 1.
    xmu = approx(np.mean(X, axis=0), abs=1e-4) == np.zeros(X.shape[-1])
    xstd = approx(np.std(X, axis=0), rel=1e-4) == np.ones(X.shape[-1])
    return ymu and ystd and xmu and xstd


def compare_dataset(d1, d2):
    (X1_train, Y1_train), (X1_test, Y1_test) = d1
    (X2_train, Y2_train), (X2_test, Y2_test) = d2
    xtrain = approx(X1_train, rel=1e-12) == X2_train
    xtest = approx(X1_test, rel=1e-12) == X2_test
    ytrain = approx(Y1_train, rel=1e-12) == Y2_train
    ytest = approx(Y1_test, rel=1e-12) == Y2_test
    return xtrain and xtest and ytrain and ytest


def test_normalized():
    datasets = ["Naval"]
    for dataset in datasets:
        assert is_normalized(dataset)


def test_seeds():
    datasets = ["Naval"]
    for dataset in datasets:
        np.random.seed(0)
        d1 = load_bb(dataset, split=0, prop=.8)
        np.random.seed(12)
        d2 = load_bb(dataset, split=0, prop=.8)
        assert compare_dataset(d1, d2)




