from __future__ import annotations
from typing import Optional, Type
import numpy as np
import logging

from ..base.base_estimator import BaseEstimator
from ..metrics import DistanceMetric, Euclidean, Manhattan, Cosine
from ..utils.validation import check_X_y, check_array, check_is_fitted

logger = logging.getLogger(__name__)

class KNNClassifier(BaseEstimator):
    """K-nearest neighbors classifier."""

    def __init__(self, k: int = 3, metric: Type[DistanceMetric] = Euclidean):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.metric_cls = metric

        self._metric: Optional[DistanceMetric] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":      
        X,y = check_X_y(X, y)
        
        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} is greater than n_samples={X.shape[0]}")

        self.X_train = X
        self.classes_, self.y_train = np.unique(y, return_inverse=True)
        self._metric = self.metric_cls()

        logger.info("Fitted KNN with %d samples and %d features", X.shape[0], X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["X_train", "y_train", "_metric"])
        X = check_array(X)

        dists = self._metric.compute(self.X_train, X)
        y_pred = []
        for i in range(dists.shape[0]):
            idx = np.argsort(dists[i])[: self.k]
            nearest = self.y_train[idx]
            counts = np.bincount(nearest, minlength=len(self.classes_))
            y_pred.append(np.argmax(counts))

        return self.classes_[np.array(y_pred)]