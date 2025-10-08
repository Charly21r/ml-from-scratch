from __future__ import annotations

import numpy as np


class KNNClassifier:
    """K-nearest neighbors classifier (from scratch)."""

    def __init__(self, k: int = 3, dist: str = "euclidean") -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}.")

        self.k = k
        self.dist = dist
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def pairwise_euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        A = self.X_train[None, :, :]
        B = X[:, None, :]
        diff = A - B
        sq = diff**2
        return np.sqrt(sq.sum(axis=-1))

    def pairwise_manhattan_distance(self, X: np.ndarray) -> np.ndarray:
        A = self.X_train[None, :, :]
        B = X[:, None, :]
        diff = np.abs(A - B)
        return diff.sum(axis=-1)

    def pairwise_cosine_distance(self, X: np.ndarray) -> np.ndarray:
        eps = 1e-12
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        X_train_norm = self.X_train / (np.linalg.norm(self.X_train, axis=1, keepdims=True) + eps)
        cos_sim = X_norm @ X_train_norm.T
        return 1 - cos_sim

    def compute_distance(self, X: np.ndarray) -> np.ndarray:
        if self.dist == "euclidean":
            return self.pairwise_euclidean_distance(X)
        if self.dist == "manhattan":
            return self.pairwise_manhattan_distance(X)
        if self.dist == "cosine":
            return self.pairwise_cosine_distance(X)
        raise ValueError(f"{self.dist!r} is not a valid distance metric")

    def _check_fitted(self) -> None:
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise RuntimeError("KNNClassifier is not fitted yet. Call 'fit(X, y)' first.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,), got shape {y.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X is empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} is greater than the number of training samples n={X.shape[0]}.")

        self.X_train = X
        self.classes_, self.y_train = np.unique(y, return_inverse=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}.")

        dists = self.compute_distance(X)
        expected_shape = (X.shape[0], self.X_train.shape[0])
        if dists.shape != expected_shape:
            raise RuntimeError(
                f"Distance matrix shape mismatch: expected {expected_shape}, got {dists.shape}."
            )

        preds = []
        for i in range(dists.shape[0]):
            nn_idx = np.argsort(dists[i])[: self.k]
            nn_labels = self.y_train[nn_idx]
            counts = np.bincount(nn_labels, minlength=len(self.classes_))
            preds.append(np.argmax(counts))

        preds_int = np.array(preds, dtype=int)
        return self.classes_[preds_int]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y have different number of rows")
        return np.mean(y_pred == y)
