import numpy as np
class NeareastCentroid:
    def __init__(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array([
            X[y==c].mean(axis=0) for c in self.classes
        ])

    def predict(self, X):
        distances = np.array([
            np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids
        ])
        return self.classes[distances.argmin(axis=0)]
    
rng = np.random.RandomState(42)
X_classes_0 = rng.randn(100, 2) + np.array([1.0 , 1.0])
X_classes_1 = rng.randn(100, 2) + np.array([-1.0, -1.0])
X = np.vstack([X_classes_0, X_classes_1])

y = np.array([0] * 100 + [1] * 100)


naive_predictor = NeareastCentroid(X, y)
print(naive_predictor.centroids)
print(naive_predictor.predict(X))