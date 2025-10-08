import numpy as np


def check_X_y(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    return X, y


def check_array(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    return X