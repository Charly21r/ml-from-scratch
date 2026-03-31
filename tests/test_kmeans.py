# import required modules
import numpy as np
import pytest
from mlfs.clustering.kmeans import KMeans

def make_simple_clusters():
	# Two well-separated clusters
	X1 = np.random.randn(10, 2) + np.array([5, 5])
	X2 = np.random.randn(10, 2) + np.array([-5, -5])
	return np.vstack([X1, X2])

def test_kmeans_fit_predict_basic():
	np.random.seed(0)
	X = make_simple_clusters()
	kmeans = KMeans(n_clusters=2, random_state=0)
	kmeans.fit(X)
	labels = kmeans.predict(X)
	# Should assign two clusters
	assert set(labels) == {0, 1}
	assert kmeans.centroids_.shape == (2, 2)

def test_kmeans_centroid_convergence():
	np.random.seed(1)
	X = make_simple_clusters()
	kmeans = KMeans(n_clusters=2, random_state=1, max_iter=100)
	kmeans.fit(X)
	# Fit again and centroids should be the same (with same random_state)
	kmeans2 = KMeans(n_clusters=2, random_state=1, max_iter=100)
	kmeans2.fit(X)
	np.testing.assert_allclose(kmeans.centroids_, kmeans2.centroids_)

def test_kmeans_score():
	np.random.seed(2)
	X = make_simple_clusters()
	kmeans = KMeans(n_clusters=2, random_state=2)
	kmeans.fit(X)
	score = kmeans.score(X)
	assert isinstance(score, float)
	# Score should be negative
	assert score < 0

def test_kmeans_one_cluster_error():
	X = np.random.randn(5, 2)
	with pytest.raises(ValueError):
		KMeans(n_clusters=1).fit(X)

def test_kmeans_more_clusters_than_samples():
	X = np.random.randn(3, 2)
	with pytest.raises(ValueError):
		KMeans(n_clusters=5).fit(X)

def test_kmeans_identical_points():
	X = np.ones((10, 2))
	kmeans = KMeans(n_clusters=2, random_state=0)
	kmeans.fit(X)
	# All centroids should be the same
	assert np.allclose(kmeans.centroids_[0], kmeans.centroids_[1])
	# All labels should be 0 (or 1, depending on implementation)
	assert len(set(kmeans.labels_)) == 1

def test_kmeans_random_state_reproducibility():
	X = make_simple_clusters()
	kmeans1 = KMeans(n_clusters=2, random_state=42)
	kmeans2 = KMeans(n_clusters=2, random_state=42)
	kmeans1.fit(X)
	kmeans2.fit(X)
	np.testing.assert_allclose(kmeans1.centroids_, kmeans2.centroids_)
	np.testing.assert_array_equal(kmeans1.labels_, kmeans2.labels_)
