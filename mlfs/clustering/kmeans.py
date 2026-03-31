from __future__ import annotations

import logging

import numpy as np

from ..base.base_estimator import BaseEstimator
from ..metrics import Euclidean
from ..utils.validation import check_array, check_is_fitted

logger = logging.getLogger(__name__)

class KMeans(BaseEstimator):
    """KMeans Algorithm"""
    def __init__(
        self,
        n_clusters: int = 8, 
        max_iter: int = 300, 
        random_state: int =42,
        tol: float = 1e-4
    ):        
        if n_clusters <= 1:
            raise ValueError(f"k must be greater than 1, got {n_clusters}")
        
        self.n_clusters = n_clusters
        self.max_iter: int = max_iter
        self.random_state: int = random_state
        self.dist = Euclidean()
        self.centroids_ = None
        self.labels_ = None

    
    def fit(self, X: np.ndarray) -> KMeans:
        X = check_array(X)

        # Create a random generator with a fixed seed
        rng = np.random.default_rng(seed=self.random_state)

        # Initialize centroids randomly
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        centroids = X[indices].copy()

        for _ in range(self.max_iter):
            # Compute the distance
            H = self.dist(centroids, X) # The order of the arguments is important here

            # Assign centroids (labels)
            labels = np.argmin(H, axis=1)

            # Update centroids
            for i in range(self.n_clusters):
                centroids[i] = X[labels==i].mean(axis=0)
        
        self.centroids_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["centroids_"])

        X = check_array(X)

        H = self.dist(self.centroids_, X)
        labels = np.argmin(H, axis=1)

        return labels

    def score(self, X: np.ndarray) -> float:
        """Return the negative inertia on the given test data."""
        H = self.dist(X, self.centroids_)
        return -np.sum(np.min(H, axis=1))
