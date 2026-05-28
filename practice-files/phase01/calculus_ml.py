"""
Here we will practice the calulus sub phase of math-foundation phase...
"""
import math
import numpy as np
import random

# defining the numerial derivative:

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-5, -2, 0 ,5, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2*x
    print(f"x={x:2d} f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")

# Numerical and analytical results match closely.
# Small differences come from floating-point approximation.

## Partial derivates and gradients:

def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        # print(point_plus, point_minus)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1, 2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")


# gradient descent to find minimum of F(x) = X square
x = 5.0
lr = 0.01
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    #print(f"step {step:2d} x={x:8.4f} f(x) = {x**2:10.6f}")
# starting at x = 5 each step moves close to x=0 minimum

### gradient descent on a 2D Function.

def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1

for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0:
        print(f"step {step:2d} point={point[0]:7.4f}, {point[1]:7.4f} f={loss:.6f}")

## comparing numerical and analytical derivative

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)

for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6} {ana:12.6f} {err:12.2e}")

## computing the Hessian Numerically

def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle hessian: {H_saddle}")  # eigen values 2, -2(mixed signs) saddle point
print(f"Bowl hessian: {H_bowl}")  # eigen values 2, 2 positive definite, local minima(0, 0)

### Taylor approximation applied

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0

for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f} sin(h)={true_val:.4f} order1={t1:.4f} order2={t2:.4f}")


## putting all together for NN
print("-" * 50)


random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2  # calculating error (y^ - y)**2 and accumumlating
        dw += 2 * error * x  # diffrentiating w.r.t w and accumulating from all samples
        db += 2 * error  # differentiating w.r.t b and accumulating from all samples
    dw /= len(xs)  # averaging gradient
    db /= len(xs)  # averaging gradient
    total_loss /= len(xs)  # averaging loss (mse)
    w -= lr * dw   ## batch gradient
    b -= lr * dw  ## batch gradient
    if epoch % 40 == 0 or epoch ==199:
        print(f"epoch {epoch:3d} w={w:.4f} b={b:.4f} loss={total_loss:.6f}")
    
print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")

## doing same with numpy

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.rand(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w*x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db
    #if epoch % 40 == 0 or epoch == 199 :
    print(f"learned : y = {w:.2f}x + {b:.2f}")


print("-" * 20 , "exerices", "-" * 20)

#exerice 1: Implement numerical_second_derivative(f, x) using numerical_derivative called twice. 
#Verify that the second derivative of x^3 at x=2 is 12.


def cubic(x):
    return x ** 3

def first_derivative(t):
    return numerical_derivative(cubic, t)


for i in [2.0, 3.0, 4.0]:
    second_drivative = numerical_derivative(first_derivative, i)
    print(f"second drivate of {i} is {second_drivative}")

# exercise 2: Use gradient descent to find the minimum of f(x, y) = (x - 3)^2 + (y + 1)^2. 
# Start from (0, 0). The answer should converge to (3, -1).
print("-"* 30)
def curve(point):
    x, y = point
    return (x - 3) ** 2 + (y + 1) ** 2

point = [0.0, 0.0]
lr = 0.1

for i in range(30):
    grad = numerical_gradient(curve, point)  # calculagin gradient
    point = [p - lr * g for p, g in zip(point, grad)]  # gradient descent
    loss = curve(point)  # min of f(x, y)
    if i % 5 == 0:
        print(f"step {i:2d} point=({point[0]:7.4f}, {point[1]:7.4f}) f = {loss:.6f}")

# this loss is coming towards the tthe point (3, -1), where this is most minimmum.
# at epoch 25 point was (2.9909, -0.9970)

## Exercise 3" Add momentum to the gradient descent loop: maintain a velocity 
# vector that accumulates past gradients. Compare convergence speed with and 
# without momentum on f(x) = x^4 - 3x^2.

print("-"* 50)

def f(x):
    return x ** 4 - 3 * x ** 2

v = 0.0
lr = 0.01
# instead of x = x -ita(g) we keep velocity v 
# so update become v = beta(v) - ita(g)
# so x = x + v, gradient is already inside v.

x = 2.0
for step in range(50):
    grad = 4*x**3 - 6*x
    v = 0.9*v - lr * grad
    x += v
    loss = f(x)
    if step % 5 == 0:
        print(f"step ={step:2d}, gradient= {grad:.6f}, v={v:.6f}, loss = {loss:.6f}")
