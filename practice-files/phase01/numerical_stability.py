'''
Here i practice the numerical stability of numbers and how to prevent them from overflow annd underflows.
Learnt the practice of precision in each datatype float32, float16, float64 etc. Also learnt how to do mixed precision training.
These practice below is important as we play with exp() log ,gradients almost always in neural network. And it's customary 
to keep in mind the numerical stability to be able to debug our nan's, inf etc so that our training goes smooth.


'''

import numpy as np
import math

# Implementing Navie vs stable softmax

def softmax_naive(logits):
    exps = [math.exp(logit) for logit in logits]
    total = sum(exps)
    return [exp/total for exp in exps]

def softmax_stable(logits):
    shifted = [z - max(logits) for z in logits]
    exps = [math.exp(e) for e in shifted]
    total = sum(exps)
    return [e/ total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [150.0, 129.0, 199.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
print(f"Naive: {softmax_naive(dangerous_logits)}")  # it's not yielding Nan Because by default number are float64 in python
#print(type([dangerous_logits[0]]))

# implement LOG-SUM-EXP

def logsumexp_naive(values):
    return np.log(sum(np.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + np.log(sum(np.exp( v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = np.array([500.0, 501.0, 502.0], dtype=np.float32)
print(f"Stable: {logsumexp_stable(large):.6f}")
#print(f"Naive logsumb: {logsumexp_naive(large)}")  # it breaks and return inf because float 32 cannot handles np.exp(500) and it 
# overflows over 3.4e38.


## Stable crossentropy

def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    shifted = [z - max(logits) for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1

print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")  # becaues by default python uses double float64 precision.

## Gradient checking:

def num_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom

        status = "OK" if rel_error < tolerance else "FAIL"
        print(f" param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")
        

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)

## Mixed precision Simulation



## Exercise1: Catastrophic cancellation. Compute the variance of [1000000.0, 1000001.0, 1000002.0] using the naive 
# formula E[x^2] - E[x]^2 in float32. Then compute it using Welford's online algorithm. Compare the errors against 
# the true variance (0.6667).

def variance_naive(values):
    #length = len(values)
    mean_x = np.mean(values)
    mean_x_sq = np.mean(values ** 2)
    return mean_x_sq - (mean_x ** 2)

def welford_approximation(values):
    n = np.float32(0.0)
    mean = np.float32(0.0)
    m2 = np.float32(0.0)

    for x in values:
        n += 1.0
        delta1 = x - mean
        mean += delta1 / n
        delta2 = x - mean
        m2 += delta1 * delta2
    
    if n < 2:
        return np.float32(0.0)
    return m2/n

var = np.array([1000000.0, 1000001.0, 1000002.0], dtype=np.float32)
true_var = 0.6667

naive_var = variance_naive(var)
welford_var = welford_approximation(var)
print(f" Naive variance Result: {naive_var}")
print(f"Welford variance result: {welford_var}") 

# calculating relative errors
rel_error_naive = abs(true_var - naive_var) / true_var
rel_error_welford = abs(true_var - welford_var) / true_var

print(f"Naive Relative Error:   {rel_error_naive:.2%}")
print(f"Welford Relative Error: {rel_error_welford:.2%}")

## Exercise2: Precision hunt. Find the smallest positive float32 value x such that 1.0 + x == 1.0 in Python. 
# This is the machine epsilon. Verify it matches numpy.finfo(numpy.float32).eps.
x = np.float32(1.0)  # starting with 1.0

while np.float32(x / 2.0) + np.float32(1.0) > np.float32(1.0):
    x = np.float32(x / 2.0)
    #print(x)

print(f"Hunted x = {x}")

print(f"official numpy epsilon: {np.finfo(np.float32).eps}")

## Exercise 3: Log-sum-exp edge cases. Test your logsumexp_stable function with: (a) all values equal, (b) one value much larger 
# than the rest, (c) all values very negative (-1000). Verify it gives correct results where the naive version fails.
val_equal = [3, 3, 3, 3]
all_larger = [1, 2, 1.5, 500.0]
all_negative = [-1000, -1400, -3500, -2100]

equal_logsumexp_stable = logsumexp_stable(val_equal)
larger_logsumexp_stable = logsumexp_stable(all_larger)
negative_logsumexp_stable = logsumexp_stable(all_negative)

equal_logsumexp_naive = logsumexp_naive(val_equal)
larger_logsumexp_naive = logsumexp_naive(all_larger)
negative_logsumexp_naive = logsumexp_naive(all_negative)

print(f"for stable logsumexp for equal, larger, negative respectively: {equal_logsumexp_stable}, {larger_logsumexp_stable}, {negative_logsumexp_stable}")
print(f"for naive logsumexp for equal, larger, negative respectively: {equal_logsumexp_naive}, {larger_logsumexp_naive}, {negative_logsumexp_naive}")
## naive logsumexp fails for all negative values. it goes to negative infinity

## Exercise 4: Gradient checking a neural network layer. Implement a single linear layer y = Wx + b and its analytical backward pass. 
# Use numerical_gradient to verify correctness for a 3x2 weight matrix.

x = np.array([3.0, 2.0], dtype=np.float32)
W = np.array([[1.0, 2.0],[4.0, 6.0],[7.0, 9.0]], dtype=np.float64)

b = np.array([0.1, 0.2, 0.1], dtype=np.float64)



# flattening all inputs 
params_flat = np.concatenate([W.ravel(), x.ravel(), b.ravel()])

def loss_func(p_flat):
    # reconstructing
    W_curr = p_flat[0:6].reshape(3, 2)
    x_curr = p_flat[6:8]
    b_curr = p_flat[8:11]

    y = np.dot(W_curr, x_curr) + b_curr

    return np.sum(y)

def numerical_gradient(f, flat_params, h=1e-5):
    grad = np.zeros_like(flat_params)
    for i in range(len(flat_params)):
        p_plus = flat_params.copy()
        p_minus = flat_params.copy()

        p_plus[i] += h
        p_minus[i] -= h

        grad[i] = (f(p_plus) - f(p_minus)) / (2 * h)
    return grad

#--5. Analytical Backward pass calculus
# dl/dy = [1, 1, 1]
dloss_dy = np.ones(3, dtype=np.float64)

dW_analytical = np.outer(dloss_dy, x)  # derivate w.r.t W(shape 3 X 2)
dx_analytical = np.dot(W.T, dloss_dy)  # derivate w.r.t x(shape (2,))
db_analytical = dloss_dy               # derivate w.r.t b(shape (3,))

# flatten analytical results in exact same seq to compare
analytical_grad_flat = np.concatenate([
    dW_analytical.ravel(),
    dx_analytical.ravel(),
    db_analytical.ravel()
])

numerical_grad_flat = numerical_gradient(loss_func, params_flat)

print("gradientt checking")

#check_gradient(analytical_grad_flat, numerical_grad_flat)
print(numerical_grad_flat)


## Exercise 5: Loss scaling experiment. Simulate training with float16: create random gradients in the 
# range [1e-9, 1e-3], convert to float16, and measure what fraction become zero. 
# Then apply loss scaling (multiply by 1024), convert to float16, scale back, and measure the zero fraction again.
np.random.seed(42)
rand_grads = np.random.uniform(1e-9, 1e-3, size=(10000,))  # good size of gradients
rand_grads_16 = np.float16(rand_grads)  # naive float16 conversion

naive_zero_fraction = np.mean(rand_grads_16 == 0.0)

# loss scaling strategy

scale_factor = 1024
scaled_grads = rand_grads * scale_factor
scaled_grads_16 = np.float16(scaled_grads)

unscaled_grads = scaled_grads_16 / np.float16(scale_factor)

scaled_zero_fraction = np.mean(unscaled_grads == 0.0)

print(f"Naive float16 zero fraction:        {naive_zero_fraction:.2%}")
print(f"Loss-scaled float16 zero fraction: {scaled_zero_fraction:.2%}")

