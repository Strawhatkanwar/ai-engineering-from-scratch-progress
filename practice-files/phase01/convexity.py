'''
Here i will build the intuition of convex function because they almost always land at global minimum. So we will check for convexity
using definition, second derivative and Hessian criteria. Implement newton method annd compare it quadretic convergence against 
gradient descent
Solve constrained optimization using lagrange methods and KKT for L1 and L2 constraint.
'''
import random
import math
import numpy as np

## Exercise 1: Convexity gallery. Test these functions for convexity using the checker: f(x) = x^4, f(x) = sin(x), f(x,y) = x^2 + y^2, 
# f(x,y) = x*y, f(x) = max(x, 0). Explain why each result makes sense.

def check_convexity(f, dim, bounds=(-5, 5), samples=1000):
    voilations = 0
    for _ in range(samples):
        x = [random.uniform(*bounds) for _ in range(dim)]
        y = [random.uniform(*bounds) for _ in range(dim)]
        t = random.uniform(0, 1)

        mid = [t * xi + (1 - t) * yi for xi, yi in zip(x, y)]
        lhs = f(*mid)
        rhs = t * f(*x) + (1 - t) * f(*y)
        if lhs > rhs + 1e-10:
            voilations += 1
    return voilations == 0, voilations

def fp(x):
    return math.pow(x, 4)

def f1(x):
    return math.sin(x)

def f2d(x, y):
    return x**2 + y**2
def f3(x, y):
    return x * y
def f4(x):
    return max(x, 0)

convexity_f = check_convexity(fp, 1)
print(f"f(x) = x^4 is convex? :{convexity_f}")
print(f"f(x) = sin(x) is convex: {check_convexity(f1, 1)}")
print(f"f(x, y)=x^2 + y^2 is Convex? {check_convexity(f2d, 2)}")
print(f"f(x, y) = x * y is Convex?     {check_convexity(f3, 2)}")
print(f"f(x) = max(x, 0) is convex? : {check_convexity(f4, 1)}")


## Exercise 2: Newton vs gradient descent race. Run both methods on f(x,y) = 50*x^2 + y^2 from the starting point (10, 10). 
# How many steps does each need to reach loss < 1e-10? 
# What happens to gradient descent when the condition number (ratio of largest to smallest Hessian eigenvalue) increases?

def f(points):
    x, y = points
    return 50.0 * x ** 2.0 + y**2

def grad_f(points):
    x, y = points
    return np.array([100.0* x, 2.0 * y])

def hessian_f(points):
    return np.array([[100.0, 0.0],
            [0.0, 2.0]])

start = np.array([10.0, 10.0])
tol=1e-12
def newtons_methods(points, tol):
    x = points.copy()
    steps = 0
    while f(x) >= tol:
        g = grad_f(x)
        H = hessian_f(x)
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if abs(det) < 1e-15:
            break

        H_inv = np.array([
            [H[1][1] / det , -H[0][1]/ det],
            [-H[1][0]/ det , H[0][0] / det],
        ])
        # newtonn step x_next = x - H^(-1) * g
        dx = H_inv @ g

        x -= dx
        steps += 1
        if steps > 100: break

    return steps, x
    
# first gradient descent

def grad_descent(points, lr, tol):
    x = start.copy()
    steps = 0
    while f(x) >= tol:
        g = grad_f(x)
        x -= lr * g
        steps += 1
        if steps > 10000: break
    return steps, x

lr = 0.01
n_steps, n_pos = newtons_methods(start, tol)
gd_steps, gd_pos = grad_descent(start, lr, tol)
print(f"newton methods converged in {n_steps} steps and final loss : {n_pos}")
print(f"GD converged in {gd_steps} and final loss {gd_pos}")
# newton methods which is a second order derivative converged the point in one step only to 0, 0, becausee looks at both 
# derivative and curvature. While gradient descent too 768 steps to converge where x-coordiatoin went to 0 and y to 10e-7 near to 0.

## Exercise 3: Lagrange multiplier geometry. Minimize f(x,y) = (x-3)^2 + (y-3)^2 subject to x + 2y = 4. 
# Verify the solution by checking that the gradient of f is parallel to the gradient of g at the solution.

def fx(points):
    x, y = points
    return (x - 3) ** 2 + (y - 3) ** 2

def f_grad(points):
    x, y = points
    return np.array([2 * x - 6, 2 * y - 6])

def g_f(points):
    x, y = points
    return x + (2 * y) - 4

def g_grad(points):
    x, y = points
    return np.array([1.0, 2.0])


def lagrange_solver(start_coords, g_val, f_grad, g_grad, steps=5000, tol=1e-6, lr=0.01, lr_lambda=0.02):
    lam = 0.0
    coords = np.array(start_coords, dtype=float)
    history = []
    for step in range(steps):
        fg = f_grad(coords)
        gv = g_val(coords)
        gg = g_grad(coords)
        # coords = [
        #     xi - lr * (fgi + lam * ggi) for xi, fgi, ggi in zip(coords, fg, gg)
        # ]
        coords -= lr * (fg + lam * gg)
        lam = lam + lr_lambda * gv
        history.append((coords[:], lam, gv))
        opt_residual = fg + lam * gg
        residual_norm = np.linalg.norm(opt_residual)  # exact size of residual(error)
        if residual_norm < tol and abs(gv) < tol:
            print(f"Converged at step {step}")
            print(f"optimal coords (x, y): {coords}")
            print(f"Lagrange Multiplier(lambda): {lam:.4f}")
            print(f"Constraint equation g(x, y) = {g_val(coords):.2e}")

            # for parallelism we can check cross product or determinant of 2 gradients
            # for 2D vectors [a, b] and [c, d], parallel vectors have ad - bc = 0
            parrallelism_check = fg[0] * gg[1] - fg[1] * gg[0]
            print(f"Gradients Parrellism Verification (cross product=0): {parrallelism_check:.2e}")
            break

    return history, coords, lam

start = [1.0, 0.9]
history, final_coords, final_lam = lagrange_solver(start, g_f, f_grad, g_grad, steps=5000)

# the optimal coordinates came out to be [1.999999, 0.999998] which is close to [2.0, 1.0], the lagrange multiplier is 2.0


## Exercise 4: Regularization constraint. Implement L1-constrained optimization: minimize (x-3)^2 + (y-2)^2 subject to |x| + |y| <= 1. 
# Show that the solution has one coordinate equal to zero (sparsity from the diamond constraint).

def fn(points):
    x, y = points
    return (x - 3) ** 2 + (y - 2) ** 2

def fn_grad(points):
    x, y = points
    return np.array([2 * x - 6, 2 * y - 4])

def gx(points):
    x, y = points
    return abs(x) + abs(y) - 1

def gradg(points):
    x, y = points
    # subgradients of L1 norm: sign of each element
    dx = 1 if x > 0 else -1
    dy = 1 if y > 0 else -1
    #return np.array([dx, dy])
    return np.array([np.sign(x), np.sign(y)])

def l1_optimization(f, f_grad, g_val, grad_g, x0, lr=0.01, lr_lam=0.01, steps=1000, tol=1e-7):
    coords = np.array(x0, dtype=float)
    lam = 0.0  # starting with lagrange multiplier

    for step in range(steps):
        fg = f_grad(coords)
        gv = g_val(coords)
        gg = grad_g(coords)
        
        # moving away from gradient (coordinate descent step)
        coords -= lr * (fg + lam * gg)

        # multiplier step for inequality: clamped at 0(KKT multiplier rule)
        # if gv <= 0 (inside the diamond), lam stays or goes to 0.
        lam = max(0.0, lam + lr_lam * gv)

        # convergence check: KKT stationarity resifual
        opt_residual = fg + lam * gg
        residual_norm = np.linalg.norm(opt_residual)

        if (residual_norm < tol and abs(gv * lam) < tol) or (np.linalg.norm(fg) < tol and gv <= 0):
            print(f"Converged at step {step}")
    return coords, lam

start = [3.0, 2.0]
final_coords, final_lam = l1_optimization(fn, fn_grad, gx, gradg, start)
print("-"* 50)

print(f"Optimal coords x, y {np.round(final_coords, 5)}")
print(f"Lagrange Multiplier (lambda): {final_lam:.4f}")
print(f"constraint Boundary Evaluation g(x, y): {gx(final_coords):.5f}")

# yes the y-coordinate is nearing to zero using this L1 contraint which is also called lasso contraint. Although it's not particularl
#0. but it's in 100 points of 0.


## Exercise 5: Hessian eigenvalue analysis. Compute the Hessian of the Rosenbrock function at (1,1) and at (-1,1). Compute 
# eigenvalues at both points. What do the eigenvalues tell you about the curvature at the minimum versus far from it?

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad_rosen(x, y):
    return [-2 *(1-x)+2 * (y - x**2) * -2 * x, 2 * (y - x ** 2) ]

def hessian_mat(x, y):
    h_xx = 2 - 4*y + 12*(x**2)
    h_xy = -4*x
    h_yy = 2.0
    
    return np.array([
        [h_xx, h_xy],
        [h_xy, h_yy]
    ])

points = [(1.0, 1.0), (-1.0, 1.0)]

for x, y in points:
    H = hessian_mat(x, y)
    eigenvalues = np.linalg.eigvals(H)

    print(f" Analysis at poinnt ({x}, {y}) ----")
    print(f"Hessian matrix:\n {H}")
    print(f"Eigenvalues: {eigenvalues}")

    if np.all(eigenvalues > 0):
        print("curvature: Positive difinite ( strict local valley)")
    elif np.all(eigenvalues < 0):
        print("Curvature: Negative definite (local peak/ maximum)")
    else:
        print("Curvature: Indefinite (Saddle point)")
    print()

# for my defined funtion the eigen values for both points came out to be same[11.6568, 0.3431] so lambda 1 is 30+ time greater
# than lambda 2
# first order derivative would bounce off voilantely back and forth between the steep canyon walls along(lambda1) and would make
# very slow progress alonng lambda2.
