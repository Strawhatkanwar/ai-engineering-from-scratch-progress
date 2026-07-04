'''
This file contains these the from scratch implementation of logistic regression model:
- logistic regression from scratch for binary classification.
- Multiclass regression using softmax from scratch
- Classification metrics from scratch for 2 class and 3 classes(confusion matrix, 3 class confusion metrics)
- I applied logistic regression to synthetically generated concentric circle data and demostrate how it lags and preforms better with polynomial features.
- computed multiclass confusion matrix calculating each class precision and recall.
- Build ROC curve from scratch using trapezoidal calculation and AUC applied to my logistic regression model of synthetic data. 

'''

import random 
import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)
    
    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0
    
    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n
    
    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f" Epoch {epoch:4d} | Loss: {loss:4f} | w: {[round(w, 3) for w in self.weights]} | b: {self.bias:.3f}")

        return self
    
    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)

## softmax regression multiclass
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s-max_score) for s in scores]
        total = sum(exp_scores)
        return [s / total for s in exp_scores]
    
    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))
    
    def fit(self, X, y, epochs=1000, print_every=100):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0

            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)

            if epoch % print_every == 0:
                print(f" Epoch {epoch:4d} | Loss: {total_loss / n:.4f}") 
        return self
       
    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)



### Confustion matrix:
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 1 and int(p) == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 0 and int(p) == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 0 and int(p) == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if int(t) == 1 and int(p) == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def fpr(self):
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print("\n  Confusion Matrix:")
        print("                  Predicted")
        print("                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")



## generating random samples
random.seed(42)
N = 500
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), class 1 center: {5, 5}")
print("First 5 samples")
for i in range(5):
    print(f" Features: [{X[i][0]:2f}, {X[i][1]:.2f}], Label: {y[i]}")

split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print('\n ----------------TRAINING LOGISTIC REGRESSION---------------------')
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=2001, print_every=200)

y_pred = [model.predict(x) for x in X_test]
print("\n ------------Confusion matrix-------------------")
metrics = ClassificationMetrics(y_test, y_pred)
metrics.print_report()

## decision boundary analysis

w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2: x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample prediction near the boundary")

test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0]
]

for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f" [{point[0]}, {point[1]}] -- prob={prob:.4f}, class={pred}")

### --------------------------------------------------------------------################################

## Exercise1: Generate a dataset that is NOT linearly separable (e.g., two concentric circles). Train logistic regression and 
# observe its failure. Then add polynomial features (x1^2, x2^2, x1*x2) and train again. Show that the accuracy improves.

# generating concentrix circles data
angles = np.linspace(0, 2*np.pi, 180)

r_inner = 1.0
r_outer = 2.0

# inner circle 
x_inner = r_inner * np.cos(angles)
y_inner = r_inner * np.sin(angles)

# outer circle
x_outer = r_outer * np.cos(angles)
y_outer = r_outer * np.sin(angles)

## making labesl and zipping them together
inner_data = list(zip(x_inner, y_inner, [0]*len(angles)))
outer_data = list(zip(x_outer, y_outer, [1]* len(angles)))

combined_data = inner_data + outer_data

x_coords, y_coords, y_c = zip(*combined_data)
X_c = list(zip(x_coords, y_coords))

log_circle = LogisticRegression(n_features=2, learning_rate=0.01)
print("---------------Training the logistic regression on concentrix circle data(non-linear) --------------------")
log_circle.fit(X_c, y_c, epochs=1001, print_every=200)

print(log_circle.accuracy(X_c, y_c))


## making poly nomical features:
X_poly = []

for i in range(len(X)):
    x1 = X[i][0]
    x2 = X[i][1]

    x1_sq = x1**2
    x2_sq = x2**2
    x1_x2 = x1 * x2

    # combining the original featuer swith new ones
    new_row = [x1, x2, x1_sq, x2_sq, x1_x2]
    X_poly.append(new_row)

combined = list(zip(X_poly, y_c))
# randomly shuffle the pairs in place
random.seed(42)
random.shuffle(combined)

# uppacking them
X_shuffled, y_shuffled = zip(*combined)
X_shuffled = list(X_shuffled)
y_shuffled = list(y_shuffled)

total_samples = len(X_shuffled)
split_idx = int(0.8 * total_samples)
# training and test
X_train = X_shuffled[:split_idx]
y_train = y_shuffled[:split_idx]
X_test = X_shuffled[split_idx:]
y_test = y_shuffled[split_idx:]

print("\n ---------------------Training again with adding Polynomial featuers-----------------------------")
log_poly = LogisticRegression(n_features=5, learning_rate=0.1)
log_poly.fit(X_train, y_train, epochs=2000, print_every=200)
print(log_poly.accuracy(X_test, y_test))


## Exercise 2: Implement a multi-class confusion matrix for the 
# 3-class softmax model. Compute per-class precision and recall. Which class is hardest to classify?

def confusion_matrix(actual, prediction):
    mat = [
    [0, 0, 0],  # actual 0
    [0, 0, 0],  # actual 1
    [0, 0, 0]   # actual 2
    ]

    for act, pred in zip(actual, prediction):
        mat[act][pred] += 1

    return mat


def compute_metrics(matrix):
    print("Per class performance metrics")
    n_classes = len(matrix)
    for k in range(n_classes):
        actual_total = sum(matrix[k])
        predicted_total = sum(matrix[i][k] for i in range(n_classes))  # sum of col k
        correct = matrix[k][k]

        recall = correct / actual_total if actual_total > 0 else 0.0
        precision = correct / predicted_total if predicted_total > 0 else 0.0

        print(f"Class {k} | precision: {precision:.4f}  | Recall: {recall:.4f}")

    
random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")
predicted = [softmax_model.predict(sample) for sample in X_test_3]
conf_mat = confusion_matrix(y_test_3, predicted)
print(conf_mat)

compute_metrics(conf_mat)


## Exercise 3: Build an ROC curve from scratch. For 100 threshold values from 0 to 1, compute the true positive rate 
# and false positive rate. Calculate the AUC (area under the curve) using the trapezoidal rule.

# this exercise was done using data from first example where i use model to predict.
# formula for trapezoidal rule : t1 and t2 with corersponding value y1 and y2 A = y1+y2/ 2(t2 - t1)

test_probs = [model.predict_proba(x) for x in X_test]

# 2. Print them to see what your model is actually doing!
print("Min predicted probability:", min(test_probs))
print("Max predicted probability:", max(test_probs))


threshold = np.linspace(0, 1 , 100)

pred_total = []

for t in threshold:
    pred_t = []
    for i in range(len(X_test)):
        pred = model.predict(X_test[i], t)
        pred_t.append(pred)
    pred_total.append(pred_t)
# pred_total =  [[log_circle.predict(X_test[i], t) for i in range(len(X_test))] for t in threshold]
#print(len(X_test[0]))
tpr = []
fpr = []

for i in range(len(pred_total)):

    x = ClassificationMetrics(y_test, pred_total[i])
    tpr.append(x.recall())
    fpr.append(x.fpr())

# Add this print statement right after your metrics loop to see what's happening
print("First 5 FPRs:", fpr[:5])
print("First 5 TPRs:", tpr[:5])
print("Last 5 FPRs:", fpr[-5:])
print("Last 5 TPRs:", tpr[-5:])

sorted_pairs = sorted(zip(fpr, tpr))
plot_fpr, plot_tpr = zip(*sorted_pairs)

def auc_roc(fpr_list, tpr_list):
    area = []
    for i in range(len(fpr_list) - 1):
        num  = (tpr_list[i] + tpr_list[i+1]) / 2.0

        # trapezoidal width diff between 2 x-values. abs() ensures that width is positive
        diff = abs(fpr_list[i+1] - fpr_list[i])
        area.append(num * diff)
    return sum(area)

final_auc = auc_roc(plot_fpr, plot_tpr)
print(f"Final Calculated AUC: {final_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(plot_fpr, plot_tpr, color='darkorange', lw=2, label=f"ROC curve (AUC={final_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random Guess(AuC=0.5)")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="lower right")
plt.savefig("roc.png")

# Here we got beautiful curve touching the y-axis and AUCC of 0.9932, as our model was also very good at predict our 2 classes.
# here 1 was our positive class and we got AUC 0.99.