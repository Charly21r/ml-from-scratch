# Machine Learning From Scratch

A clean, well-tested implementation of core machine learning algorithms in pure Python + NumPy

## 🧠 Goals
This project focuses on:
- Deep understanding of ML algorithms
- Writing production-quality Python
- Reproducing scikit-learn behavior

## 📦 Installation
```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Usage
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

## 🧪 Running tests
```bash
pytest -q
```

## 📊 Benchmarks
```bash
python benchmarks/knn_vs_sklearn.py
python benchmarks/tree_vs_sklearn.py
```

## 📝 Future Work
- `DecisionTreeRegressor`
- `RandomForest`
- missing value support / robust feature engineering

---
⭐ If you find this useful, feel free to star the repo!