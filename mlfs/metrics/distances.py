from abc import ABC, abstractmethod
import numpy as np

class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""
    @abstractmethod
    def compute(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        ...

class Euclidean(DistanceMetric):
    def compute(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        diff = X_test[:, None, :] - X_train[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

class Manhattan(DistanceMetric):
    def compute(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        diff = np.abs(X_test[:, None, :] - X_train[None, :, :])
        return np.sum(diff, axis=-1)

class Cosine(DistanceMetric):
    def compute(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        eps = 1e-12
        X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
        X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + eps)
        sim = X_test_norm @ X_train_norm.T
        return 1 - sim