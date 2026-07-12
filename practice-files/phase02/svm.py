'''
Here i'm implementing SVM from scratch and experiment with various dataset to see how support vectors
are formed, How introducing gradients help svm converge with hinge loss.
'''
import math
import random
import matplotlib.pyplot as plt

def dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))


def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n

# linear svm via gradient descent

class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])
            
    
    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]

# defining kernel functions

def linear_kernel(x, z):
    return dot(x, z)

def poly_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))


def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i])+ b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors


## Exercise 1: Generate a 2D linearly separable dataset. Train your LinearSVM and identify the 
# support vectors. Verify that the support vectors are the points closest to the decision boundary.

def demo_linear_svm():
    n = 200
    X = []
    Y = []

    for _ in range(n):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-1, 1)
        y = 1 if x1 + x2 > 0.5 else -1
        X.append([x1, x2])
        Y.append(y)
    
    svm_linear = LinearSVM()
    svm_linear.fit(X, Y)
    w = svm_linear.w
    b = svm_linear.b
    print(f"the learned weights are: {w} and learned bias is {b}")
    support_vectors = find_support_vectors(X, Y, w, b)
    print(f"The found support vectors were :{support_vectors}")

# demo_linear_svm()

## Exercise 2: Vary C from 0.001 to 1000 on a noisy dataset. Plot the decision boundary for each C value. 
# Observe the transition from wide margin (underfitting) to narrow margin (overfitting).

def demo_boundary():
    n = 500
    X = []
    Y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1.5)

        y = 1 if x1 + x2 > 0.5 else -1
        X.append([x1, x2])
        Y.append(y)
    
    # c is related lambda by inverse relation ship, so c = 1/lambda
    c_values = [0.001, 1, 50, 1000]
    import numpy as np
    # for plotting
    plt.figure(figsize=(10, 8))
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    plt.scatter(X_arr[Y_arr == 1, 0], X_arr[Y_arr == 1, 1], color='blue', alpha=0.5, label='class +1')
    plt.scatter(X_arr[Y_arr == -1, 0], X_arr[Y_arr == -1, 1], color='red', alpha=0.5, label='Class -1')
    
    # x values to use for drawing the line strings across the plot
    x1_line = np.linspace(-3, 3, 100)

    for C in c_values:
        lam = 1.0 / C
        model = LinearSVM(lr = 0.01, lambda_param=lam, n_epochs=500)
        model.fit(X, Y)
        w = model.w
        b = model.b

        # prevent divisoin by 0 if w[1] is exactly 0
        if w[1] != 0:
            # calculting x2 = -(w1/w2)*x1 - b/w2
            x2_line = -(w[0] / w[1]) * x1_line - (b / w[1])
            plt.plot(x1_line, x2_line, label=f"C = {C} (lambda = {round(lam, 3)})", linewidth=2)


    plt.title("Variation of C parameters through the Linear SVM Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("svm.png")
    plt.close()    

demo_boundary()

