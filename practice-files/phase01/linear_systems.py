'''
Here i will practice system of linear eqation Ax=b, This encompasses the system of linear equations that we solve in ML, DL.
A could be matrix of columns vector so it becomes does linear combination of columns of A gives us b. or is b in column space
of A.
Here i build soution of Ax=b using Gauss elimination, lu solver, conjugate gradient and cholesky solver.
'''
import numpy as np
import time
import matplotlib.pyplot as plt

## Exercise 1:Solve the system [[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27] using your Gaussian elimination, your LU solver, 
# and np.linalg.solve. Verify all three give the same answer within floating-point tolerance.

def gauss_elimination(A, b):
    n = len(A)
    # augmmented matrix
    M = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    for i in range(n):
        
        for j in range(i+1, n):
            factor = M[j, i] / M[i, i]  # caculating the multiplier factor
            # substract the scaled row from the current row
            M[j, i:] -= factor * M[i, i:]

    # back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        # stating with rhs constant
        total = M[i, -1]

        # substract the know variables multiplied by their coefficient
        total -= np.sum(M[i, i+1:n] * x[i+1:n])

        x[i] = total / M[i, i]
    return x

def lu_decomp(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    A = A.astype(float)

    # for i in range(n):

    #     for j in range(i, n):
    #         u_sum = sum(L[i, k] * U[k, j] for k in range(i))
    #         U[i, j] = A[i, j] - u_sum

    #     # elements of lower Triangular matrix(L)
    #     for j in range(i+1, n):
    #         if U[i, i] == 0:
    #             raise ValueError("Zero pivot encountered . Standarded LU Decomposition failed")

    #         # summation term
    #         l_sum = sum(L[j, k] * U[k, i] for k in range(i))
    #         L[j, i] = (A[j, i] - l_sum) / U[i, i]
    # vectorizing the lu_decomposition to check it's speed for exercice 5
    for i in range(n):
        # Vectorized Upper matrix row calculation
        U[i, i:] = A[i, i:] - L[i, :i] @ U[:i, i:]
        
        # Vectorized Lower matrix column calculation
        if i < n - 1:
            L[i+1:, i] = (A[i+1:, i] - L[i+1:, :i] @ U[:i, i]) / U[i, i]

    return L, U
    
def lu_solver(L, U, b):
    n = len(b)
    b = b.astype(float)

    # forward substitution(solving Ly = b for y)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])

    x = np.zeros(n)  # back substitution (solve Ux=y)
    for i in range(n-1, -1, -1):
        total = y[i] - np.sum(U[i, i+1:] * x[i+1:])
        x[i] = total / U[i, i]
    return x

A = np.array([[1,2,3],[4,5,6],[7,8,10]])
b = np.array([6, 15, 27])

np_solve = np.linalg.solve(A, b)
gauss_solve = gauss_elimination(A, b)
L, U = lu_decomp(A)
lu_solve = lu_solver(L, U, b)
print(f" Solution with numpy library was {np_solve}")
print(f"The solution from our gauss elimination methods {gauss_solve}")
print(f"The solution from lu solver is {lu_solve}")

# all three solver yielded same answer of [3. -3. 3.]. That's great

## Exercise 2: Generate a 50x5 random matrix X and target y = X @ w_true + noise. Solve for w using normal equations, 
# QR (via np.linalg.qr), SVD (via np.linalg.svd), and np.linalg.lstsq. 
# Compare all four solutions. Measure the condition number of X^T X and explain how it affects which method you trust.

X = np.random.randint(1,10, size=(50, 5))
X = X.astype(float)
noise = np.random.rand(50) * 0.5

w_true = np.array([2.1, 1.5, 2.3, 1.4, 0.5])
y = X @ w_true + noise


# solving for w using normal equations(X^T * X)w = X^T * y
A_normal = X.T @ X
b_normal = X.T @ y
w_normal = gauss_elimination(A_normal, b_normal)
print(f"Normal equations: {w_normal}")

# method QR decompositionn
Q, R = np.linalg.qr(X)  # r is upper triangular square(5x5), we can use np.linalg.solve for 
w_qr = np.linalg.solve(R, Q.T @ y)
print(f"The qr dcompositon: {w_qr}")

# method svd --> w = V*S^-1 * U^T*y

U, S, Vh = np.linalg.svd(X, full_matrices=False)
S_inv = np.diag(1.0/S)

w_svd = Vh.T @ S_inv @ U.T @ y
print(f"SVD Decomposition: {w_svd}")

# method buildin least squares

w_lstq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
print(f" the lst solution is {w_lstq}")

# measuring condition number of X^T * X
cond_XTX = np.linalg.cond(X.T @ X)
cond_X = np.linalg.cond(X)
print(f"Condition number of X: {cond_X:.2f}")
print(f"Condition number of X^T*X: {cond_XTX:.2f}")

## Exercise 3: Create a nearly singular matrix by making two columns almost 
# identical (e.g., column 2 = column 1 + 1e-10 noise). Compute its condition number. Solve Ax = b with and 
# without regularization (add 0.01 I). Compare the solutions and residuals. Explain why regularization helps.

A = [[1, 5, 3], [2, -3, 7] ,[3, 2, 4]]
A = np.array(A).astype(float)
print(A)
A[:, 1] = A[:, 0] + 1e-10 * np.random.randn(3)

print(A)

x_true = np.array([2.5, -1.0, 4.0])
b = A @ x_true + np.random.randn(3) * 0.05


def cond_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
con_num = cond_number(A)

print(f"Condition number of original A: {con_num}")

# solving without regulartioin
x_naive = np.linalg.solve(A, b)
residual_naive = np.linalg.norm(A @ x_naive - b)

## without regularisation 

print(f"True x: {x_true}")
print(f"Estimated x: {x_naive}")
print(f"Residual error: {residual_naive}")

#4. with regularisation (add 0.01 I)
A_reg = A + 0.01 * np.eye(3)
x_reg = np.linalg.solve(A_reg, b)
residual_reg = np.linalg.norm(A @ x_reg - b)

print("---with Regularisation-----")
print(f"Estimated x: {x_reg}")
print(f"Residual error: {residual_reg}")
print(f"condition number of A_reg: {cond_number(A_reg):.2f}")
## without regularization the weights(estimated x) exploded to achieve giving us tiny residual error, but with regularizatoin
# we accept some error in ordder for a stable and realistic weights


## Exercise 4: Implement the conjugate gradient algorithm for a 100x100 random symmetric positive definite matrix. 
# Count how many iterations it takes to converge to tolerance 1e-8. Compare with the theoretical maximum of n iterations.

n = 100
np.random.seed(42)
B = np.random.randn(n, n)
# B.T @ B gurantees symmetry and positive semi-definiteness
# adding n * I guarantees it is strictly positive definite ()
A = B.T @ B + n * np.eye(n)

b = np.random.randn(n)  # generate a random target vector b


def conj_grad(A, b, tol=1e-8):
    n = len(b)
    x = np.zeros(n)  # initialize x as vector of zeros
    r = b - A @ x  
    p = r.copy()

    r_dot_r = np.dot(r, r)

    for i in range(1, n+1):
        # calculate matrix-vector product once per loop for efficiency

        Ap = A @ p

        # step size
        alpha = r_dot_r / np.dot(p, Ap)


        x = x + alpha * p
        r_next = r - alpha * Ap

        # checking convergence tolerance
        if np.linalg.norm(r_next) < tol:
            return x, i
        
        # calculate beta
        r_next_dot_r_next = np.dot(r_next, r_next)
        beta = r_next_dot_r_next / r_dot_r

        p = r_next + beta * p

        # prepare for next iteration

        r = r_next
        r_dot_r = r_next_dot_r_next
    return x, n
x_sol, iterations = conj_grad(A, b)

print(f"Matrix Size(n): {n}")
print(f"Iterations to converge: {iterations}")
print(f"Converged within theoretical max? {iterations <= n}")

print(x_sol.shape)

## Exercise 5: Time your Cholesky solver vs your LU solver vs np.linalg.solve on symmetric positive definite matrices 
# of size 10, 50, 200, 500. Plot the results. Verify Cholesky is roughly 2x faster than LU.

def cholesky(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    
    for k in range(n):
        # 1. Compute the diagonal element
        s = A[k, k] - L[k, :k] @ L[k, :k]
        if s <= 0:
            raise ValueError("Matrix is not positive definite")
        L[k, k] = np.sqrt(s)
        
        # 2. Vectorize the entire column below the diagonal element at once
        if k < n - 1:
            L[k+1:, k] = (A[k+1:, k] - L[k+1:, :k] @ L[k, :k]) / L[k, k]
            
    return L

sizes = [10, 50, 200, 500]
cholesky_times = []
lu_times = []

np.random.seed(42)

for n in sizes:
    B = np.random.randn(n, n)
    A = B.T @ B + n * np.eye(n)

    # benchmarking cholesky
    start = time.perf_counter()
    _ = cholesky(A)
    cholesky_times.append(time.perf_counter() - start)

    # benchmark LU
    start = time.perf_counter()
    _, _ = lu_decomp(A)
    lu_times.append(time.perf_counter() - start)

print(f"{'size (n)':<10}{'LU Time (s)':<15}{'Cholesky Time (s)':<20}{'Speedup Factor'}")
print("-" * 60)

for idx, n in enumerate(sizes):
    ratio = lu_times[idx] / cholesky_times[idx]
    print(f"{n:<10}{lu_times[idx]:<15.5f}{cholesky_times[idx]:<20.5f}{ratio:.2f}x")

plt.figure(figsize=(8, 5))
plt.plot(sizes, lu_times, label="LU Decomposition", marker='o', linewidth=2)
plt.plot(sizes, cholesky_times, label="Cholesky Decomposition", marker='s', linewidth=2)
plt.xlabel("Matrix Size (nxn)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time: LU vs. Cholesky Decomposition")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("linear_system.png")

## As the size increases the cholesky algorithm does well annd  turns out to be 2.25x faster than LU on size 500. and almost
# 2.92 times fast on size 200

