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


print("Exercise 1: Generate a 2D linearly separable dataset. Train your LinearSVM and identify the support vectors. \nVerify that the support vectors are the points closest to the decision boundary.")

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

demo_linear_svm()

print("Exercise 2: Vary C from 0.001 to 1000 on a noisy dataset. Plot the decision boundary for each C value. \nObserve the transition from wide margin (underfitting) to narrow margin (overfitting).")

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

print("Exercise 3: Create a dataset where class boundaries are circular (not linear). Show that a linear SVM fails. \nCompute the RBF kernel matrix and show that the classes become separable in the kernel-induced feature space.")

def svm_with_circular_data(n_samples=500):
    X = []
    Y = []
    # generating non-linear data as concentric circles
    for i in range(n_samples):
        theta = random.uniform(0, 2 * math.pi)

        # assigning randomly to inner circle(class 1) or outer circle(class -1)
        if random.random() > 0.5:
            r = 1.0 + random.gauss(0, 0.15)
            y = 1
        else:
            # our circle radius + small noise
            r = 3.0 + random.gauss(0, 0.25)
            y = -1
        
        # converting polar coordinates to cartesian
        x1 = r * math.cos(theta)
        x2 = r * math.sin(theta)
        X.append([x1, x2])
        Y.append(y)
    
    linear_model = LinearSVM(lr=0.001, lambda_param=0.01, n_epochs=500)  
    linear_model.fit(X, Y)  # fitting the model

    predictios = linear_model.predict(X)
    correct = sum([1 for i in range(n_samples) if predictios[i] == Y[i]])
    accuracy = correct / len(Y)
    print(f"accuracy using the linear SVM on circular data: {accuracy * 100:.2f}%")

    # now checking by rbf transformation
    center = [0.0, 0.0]
    X_transformed = []
    for row in X:
        z_value = rbf_kernel(row, center, gamma=0.5)
        # appending the new 3rd dim to our features
        X_transformed.append([row[0], row[1], z_value])

    rbf_linear_svm = LinearSVM(lr=0.005, lambda_param=0.01, n_epochs=1000)
    rbf_linear_svm.fit(X_transformed, Y)
    predictions_rfb = rbf_linear_svm.predict(X_transformed)
    correct_rbf = sum(1 for i in range(n_samples) if predictions_rfb[i] == Y[i])
    accuracy_rbf = correct_rbf / n_samples
    print(f"Accuracy using linear svm on RBF-transformed 3d data: {accuracy_rbf * 100:.2f}%")
svm_with_circular_data()

# we got accracy without rbf kernel as 68% and with rbf kernel's 3rd dimention as 100% seemingly clearly seperating our non-linear
# classes

print("Exercise 4: Compare hinge loss vs logistic loss on the same dataset. Train a linear SVM and logistic regression.\nCount how many training points contribute to each model's decision boundary (support vectors vs all points).")


def comparison():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    data = load_breast_cancer()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data).tolist()
    y_logreg = data.target.tolist()
    y_svm = [1 if val == 1 else -1 for val in y_logreg]

    # y = [1 for i in range(len(y)) if y[i] == 1 else -1]
    linear_svm = LinearSVM(lr=0.001, lambda_param=0.5, n_epochs=1000)
    linear_svm.fit(X_scaled, y_svm)
    w = linear_svm.w
    b = linear_svm.b
    support_vectors = find_support_vectors(X_scaled, y_svm, w, b)
    
    log_model = LogisticRegression(max_iter=10000, penalty="l2", C=20.0)
    log_model.fit(X_scaled, y_logreg)
    des = log_model.decision_function(X_scaled)
    acti_logreg_points = sum(1 for score in des if abs(score) < 5.0)
    print("\n Results")
    print(support_vectors, len(support_vectors))
    print(acti_logreg_points)

comparison()

print("\nThis led to a a support vector for brest cancer dataset and 92 decision boundary points for logistic regression which models still uncertainn about")

print("_" * 25 + "Exercise 5" + "_" * 25)

print("Implement SVR (epsilon-insensitive loss). Fit it to y = sin(x) + noise. Plot the epsilon tube around the predictions and highlight the support vectors (points outside the tube")

def data(n_samples=200):

    X_raw = sorted([random.uniform(-3, 3) for _ in range(n_samples)])
    X_poly = []
    Y = []

    for x in X_raw:
        y = math.sin(x) + random.gauss(0, 0.25)
        X_poly.append([x, x**2, x**3])
        Y.append(y)
    
    return X_raw, X_poly, Y

class SVR:
    '''
    This is the cutom svr function with gradient descent
    '''
    def __init__(self, lr = 0.005, lambda_param=0.01, epsilon=0.2, n_epochs = 2000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                prediction = dot(self.w, X[i]) + self.b
                error = y[i] - prediction

                # if point is inside the tube
                if abs(error) <= self.epsilon:
                    self.w = [wj - self.lr * self.lambda_param * wj for wj in self.w]
                
                # if piont is above the tube
                elif error > self.epsilon:
                    self.w = [wj - self.lr * (self.lambda_param * wj - X[i][j])
                              for j, wj in enumerate(self.w)]
                    
                    self.b += self.lr * 1.0
                
                # if point is below the tube prediction is too high 

                else:
                    self.w = [wj -self.lr * (self.lambda_param * wj + X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * 1.0

    def predict(self, X):
        return [dot(self.w, x) + self.b for x in X]
    


def demo_svr():

    # get data
    X, X_poly, Y = data(n_samples=500)
    epsilon = 0.25
    svr_model = SVR(lr=0.01, lambda_param=0.1, epsilon=epsilon, n_epochs=2000)
    svr_model.fit(X_poly, Y)

    # get predictions
    predictions = svr_model.predict(X_poly)

    # finding support vectors( points that land on or outside the boundary of epsilon tube)
    support_vector_indices = []
    for i in range(len(Y)):
        error = abs(Y[i] - predictions[i])
        if error >= epsilon:
            support_vector_indices.append(i)

    # for plotting
    # sv_x = [X[i] for i in support_vector_indices]
    # sv_y = [Y[i] for i in support_vector_indices]

    plt.figure(figsize=(11, 7))
    plt.scatter(X, Y, color="lightgray", label="In-tube data points", alpha=0.8)
    plt.plot(X, predictions, color="blue", linewidth=2.5, label="SVR predictions $f(x)$")

    upper_tube = [p + epsilon for p in predictions]
    lower_tube = [p - epsilon for p in predictions]

    plt.plot(X, upper_tube, color="green", linestyle="--", alpha=0.7, label="$\epsilon$-tube Bounds")
    plt.plot(X, lower_tube, color="green", linestyle="--", alpha=0.7)

    # fill the tube space
    plt.fill_between(X, lower_tube, upper_tube, color="green", alpha=0.08)

    plt.savefig("svr_fit.png")
    plt.close()

    print("SVR Training complete")
    print(f"Total points: {len(Y)}")
    print(f"support vectors outside the tube: {len(support_vector_indices)}")

demo_svr()