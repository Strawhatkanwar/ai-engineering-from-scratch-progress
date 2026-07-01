'''
Here i've build Linear regression from scratch using both both norm solution as well as with weights initialization.
I have build, Multiple Linear regression, polynomial regression, lasso regression and Ridge regression
Here i have tested Stochastic gradient descent, Batch Gradient Descent, Mini-batch gradient descent on california housing dataset.
'''

import random
import math

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]
    
    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost
    
    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum((pred - actual) for pred, actual in zip(predictions, y))
        return dw, db
    
    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f" Epoch {epoch:4d} | Cost: {cost:4f} | w: {self.w:.4f} | {self.b:.4f}")

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual  in y)
        return 1 - (ss_res / ss_tot)


class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self
    
    def predict(self, X):
        return [self.w * x + self.b for x in X]
    
    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res/ss_tot)
    
# print("\n --------- Normal Equation (Closed-Form) ----------------")
# model_normal = LinearRegressionNormal()
# model_normal.fit(X, y)
# print(f"Learned: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
# print(f"R-squared: {model_normal.r_squared(X, y):.4f}")

class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
    
    def predict(self, X):
        return [self.predict_single(x) for x in X]
    
    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
    
    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f" Epoch {epoch:4f} | cost: {cost:.4f}")
        return self
    
    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean)** 2 for actual in y)
        return 1 - (ss_res / ss_tot)
    

random.seed(42)
N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)


def standardize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(len(X)):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds

y_mean_val = sum(y_multi) / len(y_multi)
y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]

X_scaled, x_means, x_stds = standardize(X_multi)

print("\n=== Multiple Linear Regression (3 features) ===")
print("Features: house size, bedrooms, age")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nWeights (standardized): {[round(w, 4) for w in multi_model.weights]}")
print(f"Bias (standardized): {multi_model.bias:.4f}")
print(f"R-squared: {multi_model.r_squared(X_scaled, y_scaled):.4f}")


## Polynomial regression

class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]
    
    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
    
    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f" Epoch {epoch:4d} | Cost: {cost:.6f}")
        return self
    
    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)
    


## Ridge regressoin

class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha= 1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
    
    def predict(self, X):
        return [self.predict_single(x) for x in X]
    
    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])

        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(yi ** 2 for yi in errors) / n
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f" Epoch {epoch:4d} | Cost: {cost:4f} | L2 Penalty: {reg_term:.4f}")
        return self
    

print("\n -------RIDGE REGRESSION ---------------------")

# print("Using same data as we used in mulitple regression--------------------------")
# ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
# ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=100)
# print(f"\nRidge Weights: {[round(w, 4) for w in ridge.weights]}")
# print(f"Plain weights: {[round(w, 4) for w in multi_model.weights]}")


## doing with scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np

np.random.seed(42)

X_sk = np.random.uniform(0, 10, (100, 1))
y_sk = 3.0 + X_sk.squeeze() + 7.0 + np.random.normal(0, 2.0, 100)


## Exercise 1: Implement batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent. 
# Compare convergence speed on the same dataset. Which converges fastest? Which has the smoothest cost curve?


def sgd(X, y, lr=0.00001, epochs = 1001):
    n_features = len(X[0])
    weights = [0.0] * len(X[0])
    bias = 0.0
    cost_history=[]
    for epoch in range(epochs):
        running_sq_error = 0.0
        for step in range(len(X)):
            row_index = random.randint(0, len(X)- 1)
            row = X[row_index]
            actual_y = y[row_index]
            single_prediction = sum(wi * xi for wi, xi in zip(weights, row)) + bias
            error = single_prediction - actual_y
            running_sq_error += error ** 2
            for j in range(n_features):
                grad_w = 2 * error * row[j]  
                weights[j] -= lr * grad_w

            grad_b = 2 * error
            bias -= lr * grad_b
        mse = running_sq_error / len(X) # epoch mse
        cost_history.append(mse)
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | cost: {mse:.4f} | weights snipped: {[round(w, 2) for w in weights[:3]]}")
    
    return weights, bias, cost_history


def mini_batch_gd(X, y, epochs=1001, lr=0.000005):
    n_samples = len(X)
    n_features = len(X[0])
    weights = [0.0] * len(X[0])
    bias = 0.0
    cost_history = []
    for epoch in range(epochs):
        indices = list(range(n_samples))
        random.shuffle(indices)
        total_squared_error = 0.0
        for i in range(0, n_samples, 16):
            batch_indices = indices[i: i + 16]
            if len(batch_indices) < 16:
                continue    
            X_batch = [X[idx] for idx in batch_indices]
            y_batch = [y[idx] for idx in batch_indices]
            if len(X_batch) < 16:
                continue
            predictions_batch = [sum(wi * xi for wi, xi in zip(weights, row)) + bias for row in X_batch]

            errors = [pred - actual for pred, actual in zip(predictions_batch, y_batch)]
            total_squared_error += sum(e ** 2 for e in errors)
            # calculating gradients by column
            for j in range(n_features):
                grad_w = (2 / 16) * sum(errors[idx] * X_batch[idx][j] for idx in range(len(X_batch)))
                weights[j] -= lr * grad_w
            grad_b = (2/16)  * sum(errors)
            bias -= lr * grad_b
        epoch_cost = total_squared_error / (i + 16 if i + 16 <= n_samples else i)
        cost_history.append(epoch_cost)
        if epoch % 100 == 0:
            weights_snippet = [f"{w:.2f}" for w in weights[:3]]
            print(f"epoch {epoch:4d} | cost: {epoch_cost:.4f} | weights: {weights_snippet}")
    return weights, bias, cost_history

#batch gradient descent
def batch_gd(X, y, lr=0.005, epochs=1001, print_every=100):
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    n = len(X)
    cost_history = []
    for epoch in range(epochs):
        predictions = [sum(wi * xi for wi, xi in zip(weights, row)) + bias for row in X]
        errors = [pred - actual for pred, actual in zip(predictions, y)]
        mse = sum([e ** 2 for e in errors]) / len(X)
        for j in range(n_features):
            grad_w = (2 / n) * sum(errors[i] * X[i][j] for i in range(len(X)))
            weights[j] -= lr * grad_w
        grad_b = (2 / n) * sum(errors)
        bias -= lr * grad_b
        cost_history.append(mse)
        if epoch % print_every == 0:
            print(f"epochs : {epoch:4d} | cost : {mse:.4f} | weights3:{[round(w, 2) for w in weights[:3]]}")
    return weights, bias, cost_history


def standardized(X, y):
    X_scaled = []
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(len(X[0]))]
    std = []
    for j in range(len(X[0])):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        std.append(var ** 0.5)

    for j in range(len(X[0])):
        center = [(X[i][j] - means[j]) / std[j] for i in range(len(X))]
        X_scaled.append(center)
    X_scaled = np.array(X_scaled)
    X_scaled = X_scaled.T
    # for y_scaling
    mean_y = sum(y) / len(y)
    var_y = sum((y[i] - mean_y) ** 2 for i in range(len(y))) / len(y)
    std_y = var_y ** 0.5
    y_scaled = [(y[i] - mean_y) / std_y for i in range(len(y))]
    y_scaled = np.array(y_scaled)

    return X_scaled, y_scaled


def train_on_cali(standardized, batch_gd, mini_batch_gd, sgd):
    import matplotlib.pyplot as plt
    X, y = fetch_california_housing(return_X_y=True)
    X_scaled, y_scaled = standardized(X, y)
    weights_dict = dict()
    cost_history = []
    for gd in [batch_gd, mini_batch_gd, sgd]:
        name = gd.__name__
        # initializing the nested dict first
        weights_dict[name] = dict()
        weights, bias, history = gd(X_scaled, y_scaled)
        weights_dict[name]["weights"] = weights
        weights_dict[name]["bias"] = bias
        cost_history.append(history)

    plt.figure(figsize=(10, 8))
    plt.plot(cost_history[0], color='r', label='Batch GD')
    plt.plot(cost_history[1], color='b', label='Mini-batch GD')
    plt.plot(cost_history[2], color='y', label = 'Stochastic GD (SGD)')
    plt.title("The loss curve: batchGD vs SGD vs MiniBGD")
    plt.xlabel("Epochs")
    plt.ylabel("Mean squared Error (Cost)")
    plt.legend()
    plt.grid(True)
    plt.savefig("The_final_loss_plot.png")
    plt.close()
    return weights_dict  

learned_params = train_on_cali(standardized, batch_gd, mini_batch_gd, sgd)
## SGD with random sampling to calculate gradient's and update weights with each randomm sample converges fast in few epochs(20-30),
# and it oscilates a lot mean while stabilizing because it's updating weights with singal samples.
## It is followed by the minibatch gradient descent which is little better then batch gd, they follow the similar trajectory of loss
# curve but minibatch is little better.

## Exercise 2: Generate data from a cubic function (y = ax^3 + bx^2 + cx + d + noise). 
# Fit polynomials of degree 1, 3, and 10. Compare training R^2 and test R^2. At what degree does overfitting become obvious?

def cubic(a, b, c, d):
    # noise = random.gauss(0, 0.5)
    X_polynomial = [x / 20.0 for x in range(0, 100)]
    return X_polynomial, [a * x ** 3 + b * x ** 2 + c * x + d + random.gauss(0, 0.5) for x in X_polynomial]

X_poly, y_poly = cubic(1, 2, 1, 1)

X_poly_mean = sum(X_poly) / len(X_poly)
X_poly_std = (sum((xi - X_poly_mean) ** 2 for xi in X_poly) / len(X_poly)) ** 0.5
X_poly_norm = [(x - X_poly_mean) / X_poly_std for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi-y_poly_mean) ** 2 for yi in y_poly)/ len(y_poly)) ** 0.5
y_poly_norm = [(y - y_poly_mean) / y_poly_std for y in y_poly]

X_train, X_test, y_train, y_test = train_test_split(X_poly_norm ,y_poly_norm, test_size=0.3)

print("\n -----Polynomial regression (degree 1, degee 3 vs Degree 10)-------------")
print("True relationship: y = x^3 + 2x^2 + x + 1")

degree = [1, 3, 10]
for deg in degree:
    lr = 0.000001 if deg == 10 else 0.01
    print(f"fitting for degree {deg}")
    poly_func = PolynomialRegression(deg, learning_rate=0.01)
    poly_func.fit(X_train, y_train, epochs=2001, print_every=100)
    print(f"the r-squared for degree{deg} is {poly_func.r_squared(X_train, y_train):.4f}")
    # prediction = poly_func.predict(X_test)
    print(f"The test r-squared for degree {deg} is {poly_func.r_squared(X_test, y_test):.4f}")

# i found out that degree 10 is highly unstable and sensitive to normalization on standard  norm(dividnig by max) it was working
# fine and giving good results when i changed and standardized the X it overflowed in less than 100 epochs.

## Exercise 3: Implement Lasso regression (L1 regularization: penalty = alpha * sum(|w_i|)). Train on the multi-feature housing data. 
# Compare which weights go to zero vs Ridge. Why does L1 produce sparse solutions while L2 does not?

class LassoRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
    
    def predict(self, X):
        return [self.predict_single(x) for x in X]
    
    def fit(self, X, y, epochs=1001, print_every=100):
        n = len(X)
        n_features = len(self.weights)
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad_mse = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                if self.weights[j] > 0:
                    l1_grad = self.alpha
                elif self.weights[j] < 0:
                    l1_grad = -self.alpha
                else:
                    l1_grad = 0.0
                total_grad = grad_mse + l1_grad
                self.weights[j] -= self.lr * total_grad
            #cost_mse = sum(e ** 2 for e in errors) / len(X)
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost_mse = sum(e**2 for e in errors) / n
                reg_term = self.alpha * sum(abs(wi) for wi in self.weights)
                total_cost = cost_mse + reg_term
                weights_snipped = [round(w, 2) for w in self.weights[:3]]
                print(f"epoch : {epoch:4d} | cost: {cost_mse:.6f}, Reg term: {reg_term:.4f} | total: {total_cost} | weights: {weights_snipped}")
        return self
    
X, y = fetch_california_housing(return_X_y=True)
X_scaled, y_scaled = standardized(X, y)
n_features = len(X[0])
lasso_fit = LassoRegression(n_features, learning_rate=0.001, alpha=0.1)
print(f"Starting XXX Lasso Regressoin(L1 Norm)--------with california housing dataset with parameters{lasso_fit.lr, lasso_fit.alpha}")
lasso_fit.fit(X_scaled, y_scaled)

## alpha value of 0.1 is best for lasso to work as it pushes some of our weights to zero(2) but if i increase it over 1.0
# model stops learning in few epochs and all weighs go to 0. it's diamond like shape quishes the weights to 0 along the axes
# where atleast 1-coordinate is 0, On the other hand L2 penalty(Right) forms a cirle and our loss typically touch this circle
# along the smooth edge, keeping all the weight alive,