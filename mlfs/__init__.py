"""
mlfs - Machine Learning From Scratch

A lightweight library implementing ML algorithms using only NumPy.
"""

from .neighbors import KNNClassifier
from .tree import DecisionTreeClassifier

__all__ = ["KNNClassifier", "DecisionTreeClassifier"]