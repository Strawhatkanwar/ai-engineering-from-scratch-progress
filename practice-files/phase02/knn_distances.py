'''
Here i built distances from scratch and build the KNN from scratch using those defined distances.
- I also build it for both methodologies weightd and unweighted.
- Built kd-tree from scratch and applied it against brute force algorithm to commpre query time with multi-dimetional datasets
- Build a comparison pipeline of different distances applied on text data(imbd movie review dataset). Implemented TF-IDF from scratch
without using any libraries
'''

import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from collections import defaultdict
import time


def l2_distance(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai , bi in zip(a, b)))

def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def cosine_distance(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = sum(ai ** 2 for ai in a) ** 0.5
    norm_b = sum(bi ** 2 for bi in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cosine_sim =  dot / (norm_a * norm_b)
    return 1 - cosine_sim

def minkowiski_distance(a, b, p=2):
    if p == float('inf'):
        return max(abs(ai -bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi)**p for ai, bi in zip(a, b)) ** (1 / p)

def standardize():
    pass

## KNN Classifier and Regressor
class KNN:
    '''
    K nearest neighbour with configurabel class k, it will have distance metric 
    and optional distance weighting
    '''
    def __init__(self, k=5, dist_f=l2_distance, weighted=False, task="classification"):
        self.k = k
        self.dist_f = dist_f
        self.weighted = weighted
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        return [self._predict_one(x) for x in X]
    
    def _predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = self.dist_f(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda pair:pair[0])
        neighbours = distances[: self.k]

        if self.task == "classification":
            return self._classify(neighbours)
        return self._regress(neighbours)

    def _classify(self, neighbours):
        if self.weighted:
            votes = {}
            for dist, label in neighbours:
                w = 1 / (dist + 1e-10)
                votes[label] = votes.get(label, 0) + w
        else:
            votes = {}
            for _, label in neighbours:
                votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)
    
    def _regress(self, neighbours):
        if self.weighted:
            w_sum = 0.0
            val_sum = 0.0
            for dist, val in neighbours:
                w = 1.0 / (dist + 1e-10)
                val_sum += (w * val)
                w_sum += w
            return val_sum / w_sum if w_sum > 0 else 0.0
        return sum(val for _, val in neighbours) / len(neighbours)

print("_"*25 + "Exercies implementatoin" + "_"*25)

## Exercise1: Implement KNN classification on a 2D dataset with 3 classes. Plot the decision boundary 
# for K=1, K=5, K=15, and K=N. Observe the transition from overfitting to underfitting.

def knn_classification():
    X, y = make_classification(n_samples=500, n_features=2, n_classes=3, n_clusters_per_class=1, n_informative=2, n_redundant=0, n_repeated=0)
    X = X.tolist()
    y = y.tolist()
    X_train = X[:400]
    y_train = y[:400]
    X_test = X[400:]
    y_test = y[400:]

    # generating grid points for decision boundary

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    x_min, x_max = min(x[0] for x in X) - 1, max(x[0] for x in X) + 1
    y_min, y_max = min(x[1] for x in X) - 1, max(x[1] for x in X) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()].tolist()


    k_vals = [1, 5, 15, len(X_train)]
    for idx, k in enumerate(k_vals):
        k_model = KNN(k=k)
        k_model.fit(X_train, y_train)
        prediction = k_model.predict(X_test)
        correct = sum(1 for i in range(len(y_test)) if prediction[i] == y_test[i])
        accuracy = correct / len(y_test)
        print(f"K = {k: <3} | Test accuracy {accuracy*100:.2f}%")

        grid_preds = k_model.predict(grid_points)
        zz = np.array(grid_preds).reshape(xx.shape)

        ax = axes[idx]
        ax.contourf(xx, yy, zz, alpha=0.3, cmap='viridis')
        
        X_train_np = np.array(X_train)
        scatter = ax.scatter(X_train_np[:, 0], X_train_np[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=20)
        ax.set_title(f"K = {k} (Acc: {accuracy*100:.1f}%)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig("knn.png")
    plt.close()

knn_classification()

## Exercise 2: Generate 1000 random points in 2, 5, 10, 50, 100, and 500 dimensions. For each dimensionality,
#  compute the ratio of the maximum  pairwise distance to the minimum pairwise distance. 
# Plot the ratio vs dimensionality to visualize the curse of dimensionality.

def generate_data(n_samples=1000, dimension=2):
    X = []
    for i in range(n_samples):
        x = [random.uniform(0, 1) for _ in range(dimension)]
        X.append(x)
    return X

def pairwise_distance(X):
    distance = []
    for x in X:
        row_dist = []
        for idx in range(len(X)):
            dist = l2_distance(x, X[idx])
            row_dist.append(dist)
        distance.append(row_dist)

    return distance

def comparison(dimension, f):
    X = f(dimension=dimension)

    max_dist = -float('inf')
    min_dist = float('inf')
    
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            dist = l2_distance(X[i], X[j])

            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
 
    ratio = max_dist / min_dist
    print(f"Dimension: {dimension:<3} | Max: {max_dist:.4f} | Min: {min_dist:.4f} | ratio {ratio:.4f}")
    return ratio
n_dimension = [2, 5, 10, 50, 100]
ratio = []
for dimension in n_dimension:
    rat = comparison(dimension, generate_data)
    ratio.append(rat)

plt.plot(n_dimension, ratio, color='r', marker='o')
plt.title("The plot is plot of ratio of max min pairwise distance to num of dimensions")
plt.xlabel("Number of dimensions")
plt.ylabel("Max / Min distance Ratio")
plt.savefig("pairwise_dist.png")
plt.close()

# the curse of dimentionality is actually is in the play, more the dimensions the less is the
# ratio of max and min pariwise distance between the points, meaning that the closest points are not that close in order
# for model to correctly group them, and farthest points are also not that far compare to the closest ones, like for 2 dimentions
# the min distance is 0.0005 and max is 1.35, so that ratio is nearly 3000, meaning there's clear distinction, but for 100D,
# the minimum is 2.8 and max is 5.17 giving the ratio close to 1.8. that means the closes point is 2.8 units far to each other which
# considerable distance for closest points,

## Exercies 3: Compare L1, L2, and cosine distance for KNN on a text classification problem (use TF-IDF vectors). 
# Which metric gives the best accuracy? Why does cosine tend to win for text?

def text_classification():
    data = pd.read_csv("data/imdb_train.csv").sample(n=1000, random_state=42).reset_index(drop=True)

    words_doc_wise = []  # stores word counts per document
    doc_counts = defaultdict(int)  

    STOP_WORDS = {"the", "a", "an", "and", "or", "in", "of", "to", "is", "it", "this", "that", "was", "for", "with", "as", "by", "on", "movie", "film"}

    for doc in data["text"]:
        clean_doc = doc.lower().replace(".", "").replace(",","").replace("!", "").replace("?", "").replace("/", "")
        words = clean_doc.split(" ")

        local_counts = defaultdict(int)
        for word in words:
            if word not in STOP_WORDS and len(word) > 2:
                local_counts[word] += 1
        words_doc_wise.append(local_counts)

        for word in local_counts.keys():
            doc_counts[word] += 1

    sorted_vocab = sorted(doc_counts.items(), key=lambda item: item[1], reverse=True)[:5000]  # limiting our vocabulary from 35k to 5k
    vocab = [word for word, count in sorted_vocab]

    vocab_index = {word:i for i, word in enumerate(vocab)}
    total_docs = len(data)

    idf = {}
    for word in vocab:
        count = doc_counts[word]
        idf[word] = math.log(total_docs / count)

    # let's generate tf idf feature matrix
    X = []

    for doc_idx, local_counts in enumerate(words_doc_wise):
        vector = [0.0] * len(vocab)

        # total words in this specific documents
        doc_len = sum(local_counts.values())

        for word, freq in local_counts.items():
            if word in vocab_index:
                tf_val = freq / doc_len
                idf_val = idf[word]

                # placing the weight in exact vocabulary slot
                idx = vocab_index[word]
                vector[idx] = tf_val * idf_val

        X.append(vector)

    # target labels(y)
    y = data['label'].tolist()

    X_train = X[:800]
    y_train = y[:800]
    X_test = X[800:]
    y_test = y[800:]

    distances = [l1_distance, l2_distance, cosine_distance]

    for dist_f in distances:
        model = KNN(k = 5, dist_f=dist_f)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        correct = sum(1 for i in range(len(y_test)) if preds[i] == y_test[i])
        accuracy = correct / len(y_test)
        print(f"Accuracy with distance {dist_f.__name__} for textual data is {accuracy*100:.2f}%")


## Exercise 4: Implement a KD-tree and measure query time vs brute force for datasets of 1k, 10k, 
# and 100k points in 2D, 10D, and 50D. At what dimensionality does the KD-tree stop being faster than brute force?

class Node:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, depth=0):
        self.root = None

    def fit(self, X, y):
        self.root = self.build(X, y)

    def build(self, X, y, depth=0):
        n_samples = len(X)
        if n_samples == 0:
            return None
        
        n_features = len(X[0])

        axis = depth % n_features  # for 2d Depth 0 - axis 0 (X), Depth 1 - Axis (Y)

        # sort points by the selected axis to find median
        combined = sorted(zip(X, y), key=lambda item: item[0][axis])
        median_idx = n_samples // 2
        median_point, median_label = combined[median_idx]

        # creating node and recursively build left and right branches
        left_X = [item[0] for item in combined[:median_idx]]
        left_y = [item[1] for item in combined[:median_idx]]

        right_X = [item[0] for item in combined[median_idx + 1:]]
        right_y = [item[1] for item in combined[median_idx + 1:]]

        return Node(
            point=median_point,
            label=median_label,
            axis=axis,
            left=self.build(left_X, left_y, depth + 1),
            right=self.build(right_X, right_y, depth + 1)
        )
    
    def query(self, point, k=1):
        best = []
        self._search(self.root, point, k, best)
        best.sort(key=lambda x: x[0])
        return best

    def _search(self, node, point, k, best):
        if node is None:
            return None
        
        dist = l2_distance(point, node.point)
        if len(best) < k:
            best.append((dist, node.label, node.point))
            best.sort(key=lambda x: x[0])
        elif dist < best[-1][0]:

            best[-1] = (dist, node.label, node.point)
            best.sort(key=lambda x: x[0])

        axis = node.axis
        diff = point[axis] - node.point[axis]

        if diff <= 0:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        self._search(first, point, k, best)

        if len(best) < k or abs(diff) < best[-1][0]:
            self._search(second, point, k, best)


def brute_force_knn(X_train, query_point, k=1):
    """Calculates distance to Every point in X_train"""
    distances = []

    for i, point in enumerate(X_train):
        dist = l2_distance(point, query_point)
        distances.append((dist, point))
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def generate_classification_data(dimension, n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=dimension,
        n_informative=min(dimension, 5),
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    return X, y

dimension_size = [2, 10, 50]
sample_sizes = [1000, 10000, 100000]

def benchmark():
    print(f"{'Dim':<5} | {'samples':<8} | {'KD-tree Query':<20} | {'Brute Force':<20} | {'Winner':<12}")
    for dim in dimension_size:
        for sample in sample_sizes:
            X, y = generate_classification_data(dimension=dim, n_samples=sample)
            X = X.tolist()
            y = y.tolist()

            # making the trees
            tree = KDTree()
            tree.fit(X, y)

            query_points = [[random.gauss() for _ in range(dim)] for _ in range(100)]
            
            # timing kd-queries
            start = time.time()
            for qp in query_points:
                tree.query(qp, k=1)
            kd_time = time.time() - start

            # timing brute-force queries
            start = time.time()
            for qp in query_points:
                brute_force_knn(X, qp, k=1)
            brute_time = time.time() - start

            winner = "KD tree" if kd_time < brute_time else "Brute force"
            print(f"{dim:<5d} | {sample:<8d} | {kd_time:<20.4f}s | {brute_time:<20.4f}s | {winner:<12}")

benchmark()

## Exercise 5: Build a weighted KNN regressor for y = sin(x) + noise. Compare it with unweighted KNN for K=3, 10, 30. 
# Show that weighting produces smoother predictions, especially for large K.

def weightedknnregressor():
    n_train = 400
    X_train = [[random.uniform(-3, 3)] for i in range(n_train)]
    y_train = [math.sin(x[0]) + random.gauss(0, 0.25) for x in X_train]

    # generating dense continuous evaluation grid
    X_grid = np.linspace(-3, 3, 200)
    X_test = [[x] for x in X_grid]

    y_true = [math.sin(x[0]) for x in X_test]  # zero noise 

    k_values = [3, 10, 30]
    print(f"{'K':<5} | {'Unweighted MSE (vs True Sin)':<28} | {'Weighted MSE (vs True Sin)':<28}")
    print("_" * 65)

    for k in k_values:
        # weighted

        weightedknn = KNN(k = k, dist_f=l2_distance, weighted=True, task="regression")
        weightedknn.fit(X_train, y_train)
        preds_w = weightedknn.predict(X_test)
        error_w = sum((p - y)**2 for p, y in zip(preds_w, y_true)) / len(y_true)

        # unweighted

        unweightedknn = KNN(k = k, dist_f=l1_distance, weighted=False, task="regression")
        unweightedknn.fit(X_train, y_train)
        preds_u = unweightedknn.predict(X_test)
        error_u = sum((p - y)**2 for p, y in zip(preds_u, y_true)) / len(y_true)

        print(f" {k:<5d} | {error_w:<28.5f} |  {error_u:<28.5f}")


weightedknnregressor()

# Here weighted wins for sure even though for smaller k(3) we get errors close but less for weighted(0.021) and unweighted had(0.030)
# but for larger k(10, 30), i see considerable difference like for k=10 the weights loss is 3.1x smaller and for k=30 it's 2.7 times
# smaller. It is because in this particular data weighted KNN uses distance weighting, so distant points in 30-neighbour neighbourhood
# get virtually 0 weights, so they don't flattent the sin waves peak. but in unweighted forces distance points to have exact same vote
# as adjacent points. Near of peaks of sin(x), points on the slope pull the prediction downward toward 0, flattening the curve and 
# causing systemic error.