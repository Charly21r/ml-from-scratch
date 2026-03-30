# Machine Learning From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

A clean, well-tested, production-quality implementation of core machine learning algorithms in pure Python and NumPy. Educational and performant.

## 🎯 Goals
- **Deep Understanding**: Clean implementations showing how algorithms work internally
- **Quality**: Professional code standards matching scikit-learn
- **Educational Value**: Well-documented, tested implementations for learning

## ✨ Features

- **DecisionTreeClassifier**: CART algorithm with Gini/Entropy criteria, pruning, feature importances
- **KNNClassifier**: Flexible k-NN with multiple distance metrics
- **Comprehensive Testing**: Rigorous test suite with pytest and coverage


## 📦 Installation

### From Source

```bash
git clone https://github.com/yourusername/ml-from-scratch.git
cd ml-from-scratch
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
import numpy as np
from mlfs import DecisionTreeClassifier, KNNClassifier

# Decision Tree
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
print(clf.predict(np.array([[0, 0], [1, 1]])))
print(clf.feature_importances_)

# KNN
X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
y = np.array([0, 0, 1, 1, 1])
knn = KNNClassifier(k=3).fit(X, y)
print(knn.predict(np.array([[0.1], [1.9]])))
```

## 🧪 Testing

```bash
pytest --cov=mlfs --cov-report=html
```

## 📊 Benchmarks

```bash
python benchmarks/knn_vs_sklearn.py
python benchmarks/tree_vs_sklearn.py
```

## 📚 Documentation

- [API Standards](API_STANDARDS.md) - Design patterns


## 📄 License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.

---

⭐ If you find this useful, please star the repo!
