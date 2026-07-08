'''
Here i built tree-based methods from scratch: Decision trees and random forest
- I built criterion from scratch: gini and entropy
- Built the Decision trees for both classification and Regression
- Experimented on mutiple datasets to make comparison how Random forest reduce variance and how gini and entropy criterion
produce kind of same results for classification tasks.

'''

import math
import random
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )

## find the best split
# try every feature and every threshold. Return the one with highest information gain.

def information_gain(parent_labels, left_labels, right_labels, criterion='gini'):
    measure = gini_impurity if criterion == 'gini' else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) + 
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity


def variance_reduction(parent_values, left_values, right_values):
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0
    n = len(parent_values)
    parent_var = _variance(parent_values)
    child_var = (
        (len(left_values) / n) * _variance(left_values)
        + (len(right_values) / n) * _variance(right_values)
    )
    return parent_var - child_var


def _variance(values):
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _mean(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def majority_vote(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


## building decisiontree class

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, criterion="gini", max_features=None, task="classification"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.task = task
        self.tree = None
        self.feature_importances_ = None
        self.n_features = 0
        self.n_samples = 0

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)

        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
    
    def _build(self, X, y, depth):
        if self.task == "classification":
            all_same = len(set(y)) == 1
        else:
            all_same = len(set(y)) == 1

        if all_same:
            return {"leaf": True, "value": y[0] if self.task == "classification" else _mean(y)}

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        if len(y) < self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0:
            return self._make_leaf(y)

        left_X, left_y, right_X, right_y = self._split_data(
            X, y, best_feature, best_threshold
        )

        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)

        weight = len(y) / self.n_samples
        self.feature_importances_[best_feature] += weight * best_gain

        left_child = self._build(left_X, left_y, depth + 1)
        right_child = self._build(right_X, right_y, depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_child,
            "right": right_child,
        }
    def _make_leaf(self, y):
        if self.task == "classification":
            return {"leaf": True, "value": majority_vote(y)}
        else:
            return {"leaf": True, "value": _mean(y)}

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = -1.0

        if self.max_features is None:
            feature_indices = list(range(self.n_features))
        elif self.max_features == "sqrt":
            k = max(1, int(math.sqrt(self.n_features)))
            feature_indices = random.sample(range(self.n_features), k)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, self.n_features)
            feature_indices = random.sample(range(self.n_features), k)
        else:
            feature_indices = list(range(self.n_features))

        for feature_idx in feature_indices:
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            if len(values) <= 1:
                continue

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                if self.task == "classification":
                    gain = information_gain(y, left_y, right_y, self.criterion)
                else:
                    gain = variance_reduction(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain
    
    def _split_data(self, X, y, feature, threshold):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree
        if node["leaf"]:
            print(f"{indent}Predict: {node['value']}")
            return
        print(f"{indent}Feature {node['feature']} <= {node['threshold']:.4f}?")
        print(f"{indent}  Yes:")
        self.print_tree(node["left"], indent + "    ")
        print(f"{indent}  No:")
        self.print_tree(node["right"], indent + "    ")


def load_data():

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    tree = DecisionTree(max_depth=3, max_features=4)
    tree.fit(X_train, y_train)
    tree.print_tree()


## EXercise 1: Train a single decision tree on a 2D dataset with 3 classes. Manually trace the splits 
# and draw the rectangular decision boundaries. Compare the boundaries at max_depth=2 vs max_depth=10.

def two_d_data(max_depths):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples = 200, n_features=2, 
                      centers=3, cluster_std=2.5, random_state=42)
    X = X.tolist()
    y = y.tolist()
    for depth in max_depths:
        print(f"\n" + "=" * 40)
        print(f" TRAINING DECISION TREE (max_depth={depth})")
        print("=" * 40)

        # Initialize and fit the tree
        tree = DecisionTree(max_depth=depth)
        tree.fit(X, y)

        # Print the resulting tree hierarchy
        tree.print_tree()

two_d_data([2, 10])

## Exercise 2: Implement variance reduction splitting for regression trees. Generate y = sin(x) + noise
#  for 200 points and fit your regression tree. Plot the tree's piecewise-constant predictions against the true curve.

def data():
    X_reg = [[random.uniform(0, 2 * math.pi)] for _ in range(200)]
    y = [math.sin(x[0]) + random.normalvariate(mu=0, sigma=0.2) for x in X_reg]
    return X_reg, y

def model_fitting():

    model = DecisionTree(max_depth=3, task="regressoin")
    X_reg, y = data()
    model.fit(X_reg, y)

    X_smooth = [[i * 0.01] for i in range(int(2 * math.pi * 100))]
    y_true = [math.sin(x[0]) for x in X_smooth]
    y_pred = model.predict(X_smooth)

    # plotting
    plt.figure(figsize=(10, 6))

    plt.scatter(
        [x[0] for x in X_reg],
        y, 
        color="gray",
        alpha=0.6,
        label="Noisy data points"
    )
    plt.plot([x[0] for x in X_smooth], y_true, label="True sin(x)", color="blue")
    # plot the tree's staircase-like predcitions
    plt.step(
        [x[0] for x in X_smooth],
        y_pred,
        color="red",
        linewidth=2.5,
        label="Tree Prediction (max_depth=3)",
    )
    plt.title("Regressoin Tree piecewise-Constant Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("Reg_tree.png")
    plt.close()

model_fitting()

## Exercise 3: Build a random forest with 1, 5, 10, 50, and 200 trees. Plot training accuracy 
# and test accuracy vs number of trees. Observe that test accuracy plateaus but does not decrease (forests resist overfitting).

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                min_samples_split=2, max_features="sqrt", 
                criterion="gini", task="classification"):
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.task = task
        self.trees = []

    def fit(self, X ,y):
        self.trees = []
        n = len(X)
        for _ in range(self.n_trees):
            indices = [random.randint(0, n-1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
                task=self.task
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            if self.task == "classification":
                votes = {}
                for preds in all_preds:
                    v = preds[i]
                    votes[v] = votes.get(v, 0) + 1
                predictions.append(max(votes, key=votes.get))
            else:
                predictions.append(
                    sum(preds[i] for preds in all_preds) / len(all_preds)
                )
        return predictions

    def feature_importances(self):
        n_features = self.trees[0].n_features
        importances = [0.0] * n_features
        for tree in self.trees:
            for j in range(n_features):
                importances[j] += tree.feature_importances_[j]
        total = sum(importances)
        if total > 0:
            importances = [imp / total for imp in importances]
        return importances

def rand_forest_model(n_trees):


    X, y = load_breast_cancer(return_X_y=True)
    X = X.tolist()
    y = y.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_acc = []
    test_acc = []

    for i in n_trees:
        model = RandomForest(n_trees=i, max_depth=5, task="classification")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_train = model.predict(X_train)
        correct_train = sum([1 for i in range(len(preds_train)) if preds_train[i] == y_train[i]])
        train_accuracy = correct_train/ len(y_train)
        train_acc.append(train_accuracy)
        correct = sum([1 for i in range(len(preds)) if preds[i] == y_test[i]])
        test_accuracy = correct/ len(y_test)
        test_acc.append(test_accuracy) 
        print(f"accuracy for n_trees {i} train_accuracy is {train_accuracy} and test_accuracy is {test_accuracy}")
    
    plt.figure(figsize=(10, 8))
    plt.plot(n_trees, train_acc, label="Train Accuracy", marker="o")
    plt.plot(n_trees, test_acc, label="Test Accucary", marker="o")
    plt.xlabel("Number of trees")
    plt.ylabel("Accraucy")
    plt.title("Random forest: Accuracy vs the Number of trees")
    plt.legend()
    plt.grid(True)
    plt.savefig("random_forestvs_trees.png")
    plt.close()

rand_forest_model([1, 5, 10, 50, 200])

# test accuracy platues as number of trees grows

## Exercise 4: Compare Gini impurity vs entropy as split criteria on 5 different datasets. 
# Measure accuracy and tree depth. In most cases, they produce nearly identical results. Explain why.

def synthetic_data():
    return make_classification(
        n_samples=500, n_features=10, n_classes=3, n_informative=5
    )

datasets = [
    ("Iris", load_iris(return_X_y=True)), 
    ("Breast Cancer", load_breast_cancer(return_X_y=True)),
    ("Digits", load_digits(return_X_y=True)),
    ("Wine", load_wine(return_X_y=True)),
    ("Synthetic 3-class", synthetic_data())
    ]
criterions = ["gini", "entropy"]
def criterion_comparison(datasets, criterions):
    for name, data in datasets:
        X, y = data
        X = X.tolist()
        y = y.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        
        print(f"\n--- Dataset: {name} ---")

        for criteria in criterions:
            model = RandomForest(n_trees = 20, max_depth=5, criterion=criteria)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            correct = sum([1 for i in range(len(preds)) if preds[i] == y_test[i]])
            accuracy = correct / len(y_test)

            print(f" criteria {criteria:<7} test accuracy is {accuracy:.4f}")
        
# criterion_comparison(datasets, criterions)

# Here as we increase the tree depth accuracy does increase for somme datasets like for digits but that maybe the case because of 
# complexity and samples of dataset, my synthetic dataset does improve with tree_depth as well as the digits dataset as it has 9
# classes to classify.
#  As to why both gini and entropy produces the same kind of results maybe because the math behaves same for both of those criterions
# the only difference is that for entropy we scale by the log base 2 to calcualate information surprise and gini uses a squared
# probability approach, both metrics form nearly overlapping upside-down parabolas when plotted. The both equal 0.0 at perfect purity
# and peak at exactly 0.5 for a 50/50 binary split. Becase they rank the impurity of potential data splits in the exact same relative
# order, they select the exact same feature and threshold to split an over 95% of the time.



# Exericise 5: Implement permutation importance. Compare it with MDI importance on a dataset where one feature is random noise 
# but has high cardinality. MDI will rank the noise feature highly. Permutation importance will not.

def permutation_imp():
    n = 500
    X = []
    Y = []

    for _ in range(n):
        imp1 = random.uniform(-2, 2)
        noise1 = random.randint(1000, 9999)  # high cardinality feature
        noise2 = random.gauss(0, 1)
        imp2 = random.uniform(-1, 1)
        y = 1 if imp1 + imp2 > 1 else 0
        X.append([imp1, imp2, noise1, noise2])
        Y.append(y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = RandomForest(n_trees=20, max_depth=3)
    model.fit(X_train, y_train)

    print("\n MdI importances gini based")
    mdi_importances =  model.feature_importances()
    for idx, imp in enumerate(mdi_importances):
        print(f" features {idx}: {imp:.4f}")

    # part B calculate permutation importance
    base_preds = model.predict(X_test)
    base_accuracy = sum([1 for i in range(len(y_test)) if base_preds[i] == y_test[i]]) / len(y_test)

    permutation_importances = []
    n_features = len(X[0])

    for col_idx in range(n_features):
        X_corrupted = [row[:] for row in X_test]  # copy of our own X_test

        # extract the column values shuffle them and put them back
        col_values = [row[col_idx] for row in X_corrupted]
        random.shuffle(col_values)

        for i in range(len(X_corrupted)):
            X_corrupted[i][col_idx] = col_values[i]

        # now predict on our corruped data:
        corruped_preds = model.predict(X_corrupted)
        corrupted_acc = sum(
            [1 for i in range(len(y_test)) if corruped_preds[i] == y_test[i]]
        ) / len(y_test)

        # let's see the drop in accuracy
        importance_drop = base_accuracy - corrupted_acc
        permutation_importances.append(importance_drop)

    print("\n=== Permutation features importances")
    for idx, imp in enumerate(permutation_importances):
        print(f"Features {idx}: {imp:.4f}")

permutation_imp()

# I have feature 3(noise1) having high cardinality and noise2 as a random noise, mdi importances gives somme importance fo those features
# for high cardinal featuers(3) it gave 0.0126 and feature 4 it gave 0.0389 but permutation importance didn't give any importance to
# those featuers and straight out reject them with 0 importance.
