"""
Here i will defin and use the optimization algorithms from scratch and use them in some examples
Also i will do exercises to solidify the learning 
"""
import torch
## defining a test function

# rosenbrock function

def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]

### vanilla gradient descent

class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr
    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)] 


## SGD with momentum

class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]

## Adam class

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
        if self.v is None:
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [self.beta1 * m + (1 - self.beta1) * g
        for m, g in zip(self.m, grads)
        ]

        self.v = [self.beta2 * v + (1 - self.beta2) * g ** 2
                  for v, g in zip(self.v, grads)
                  ]
        
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
             for p, mh, vh in zip(params, m_hat, v_hat)
             ]

## running and comparing it:

def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)


for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("ADAM", adam_history)]:
    final = history[-1]
    loss = rosenbrock(final)
    print(f"{name:6s} --> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

# using above optimizer in PyTorch

model = torch.nn.Linear(784, 10)

sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001)
adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)

print("-"*25 + "Exercise" + "-"*25)

## Exercise 1: Learning rate sweep. Run vanilla gradient descent on the Rosenbrock function 
# with learning rates [0.0001, 0.0005, 0.001, 0.005, 0.01]. Plot or print the final loss 
# after 5000 steps for each. Find the largest learning rate that still converges.

lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01]


# loss_list = []

for i in lr_list:

    params_gd = [1.2, 1.0]
    gd = GradientDescent(lr = i)

    for step in range(5000):
        grads_gd = rosenbrock_gradient(params_gd)
        params_gd = gd.step(params_gd, grads_gd)
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {rosenbrock(params_gd):.6f}, params = {params_gd}")


# for lr, loss in zip(lr_list, loss_list):
#     print(f"LR: {lr:<7} | Final Loss: {loss:.6f}")


## Exercise 2: Momentum comparison. Run SGD with momentum values [0.0, 0.5, 0.9, 0.99] on the 
# Rosenbrock function. Track the loss at every step. Which momentum value converges fastest? Which overshoots?

mom_list = [0.0, 0.5, 0.9, 0.99]

for i in mom_list:
    params = [1.2, -1.1]
    gdm = SGDMomentum(momentum=i)

    start_loss = rosenbrock(params)

    for step in range(100):
        gdm_grad = rosenbrock_gradient(params)
        params = gdm.step(params, gdm_grad)
        loss = rosenbrock(params)
        if step % 10 == 0:
            print(f"Loss at step {step} is {loss}")
    print(f"start loss was {start_loss} and final loss for momentum= {i} was {loss}")
## loss coverges faster with momentum 0.90, it looks going upward in first few steps but it converges
# fast and reaches the lowest in 1000 steps of 2.45e-5
# The momentum with 0.99 overshots badly just in 10th step it was 38070.74 and in next few step
# it goes out of memory.

## exercise 3: Saddle point escape. Define the function f(x, y) = x^2 - y^2 (a saddle point at the origin).
#  Start at (0.01, 0.01). Compare how vanilla GD, SGD with momentum, and Adam behave. Which escapes the saddle point?

def f(params):
    x, y = params
    return x ** 2 - y ** 2

def f_grad(params):
    x, y = params
    df_dx = 2 * x
    df_dy = - 2 * y
    return [df_dx, df_dy]

start = (0.01, 0.01)

vgd = GradientDescent()
gdm = SGDMomentum()
adam = Adam()

for step in range(5001):
    grads = f_grad(start)
    start = vgd.step(start, grads)
    loss = f(start)
    if step % 500 == 0:
        print(f"vanilla gradient at descent at step {step} for params {start} loss is {loss}")
# # Vanilla sgd reaches [4.485e-07, 218.5120] at step 500 with loss -47747.512

for step in range(5001):
    grads = f_grad(start)
    start = gdm.step(start, grads)
    loss = f(start)
    if step % 500 == 0:
        print(f"SDG momentum at step {step} for params {start} loss is {loss}")
# # SGD with momentum reaches [7.553e-61, 1.926e35] with loss -3.711e70 at step 5000. bypasses the saddle point of 0.0.
# # The y-coordinate was maximum here because of velocity compounding. it remember 90% of its speed everytime..

for step in range(5001):
    grads = f_grad(start)
    start = adam.step(start, grads)
    loss = f(start)
    if step % 500 == 0:
        print(f"Adam at step {step} for params {start} loss is {loss}")
# adam gives [-4.469e-117, 6.832] at step 5000 with loss of -46. it bypasses the saddle point because of initial nudge(0.01, 0.01)
# adam was slow and reached 6.8 of y-coordinate because of it's mechanism to dividing learning rate by square root of historical
# gradient(sqrt(v))

## exercise 4: Implement learning rate decay. Add an exponential decay schedule to the GradientDescent class: lr = lr_0 * 0.999^step. 
# Compare convergence with and without decay on the Rosenbrock function.

# rewriting the gradient descent for this exercise

params = [1.1 , 0.02]
sgd_vanilla = GradientDescent(lr = 0.001)

for step in range(1001):
    grads = rosenbrock_gradient(params)
    params = sgd_vanilla.step(params, grads)
loss_no_decay = rosenbrock(params)

class GradientDescentDecay:
    "Class with exponential decay at every step"
    def __init__(self, lr_0=0.001):
        self.lr_0 = lr_0
        self.lr = lr_0
        self.step_count = 0

    def step(self, params, grads):
        # advancing
        self.step_count += 1

        # exponential decay formula
        self.lr = self.lr_0 * (0.999 ** self.step_count)
        return [p - self.lr * g for p, g in zip(params, grads)]

params_decay = [1.1, 0.02]
loss = rosenbrock(params)

sgd_decay = GradientDescentDecay(lr_0=0.001)
for step in range(1001):
    grads = rosenbrock_gradient(params_decay)
    params_decay = sgd_decay.step(params_decay, grads)
    loss = rosenbrock(params_decay)
    if step % 100 == 0:
        current_loss = rosenbrock(params_decay)
        print(f"Step {step:3d} | Loss: {current_loss:.6f} | current lr: {sgd_decay.lr:.6f}")
    loss_with_decay = rosenbrock(params_decay)

print("-" * 50)
print(f"Final Loss WITHOUT Decay: {loss_no_decay:.8f}")
print(f"Final Loss WITH Decay:    {loss_with_decay:.8f}")