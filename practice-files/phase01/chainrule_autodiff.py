"""
Here i will apply and implement pytorch autograd from scratch. also apply chain rule.
See how pytorch audograd works under the hood
"""
import random
import torch
## step 1: The value classs

class Value:
    def __init__(self, data, children=(), op=''):
        '''
        Every value stores it numeric data, its gradient(initially zero), a backward function,
        and pointers to child nodes that produced it.

        '''
        
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    ## step2: Artimatic operations with gradient tracking

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    #Each operation creates a closure that knows how to compute local 
    # gradients and multiply by the upstream gradient (out.grad). 
    #The += handles the case where a value is used in multiple operations

    # Step 3: Backward pass
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            '''
            Topological sort ensures every node's gradient is fully 
            computed before it propagates to its children. The seed gradient is 1.0 (dy/dy = 1).
            '''

            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        # seed the final output gradient

        self.grad = 1.0
        #2. Walk backward through the sorted list
        # print("Nodes found in graph:", [n._op if n._op else 'input' for n in reversed(topo)])
        for v in reversed(topo):
            v._backward()
    
    # Step 4: more operation for a Complete engine
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)
    
    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

# seeing examples using our class
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)

# forward pass building the chain
d = a * b # d.data become 6 
e = d * c # e.data becomes 24

# the backward pass
e.backward()
print(a)  # Value(data=2.0000, grad=12.0000)
print(b)  # Value(data=3.0000, grad=8.0000)
print(c)  # Value(data=4.0000, grad=6.0000)
print(d)  # Value(data=6.0000, grad=4.0000)
print(e)  # Value(data=24.0000, grad=1.0000)


## mini mlp from sccratch
# with our complete Value class we can build a neural network. No PyTorch. No Numpy. just
# values and the chain rule.



class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()  # for exercise 3.
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# A Neuron computes tanh(w1x1 + w2x2 + ... + b). A Layer is a list of neurons. 
# An MLP stacks layers. Every weight is a 
# Value, so calling loss.backward() propagates gradients to every parameter.

### training on XOR
random.seed(42)

model = MLP([2, 4, 1])  # 2 inputs, 4 hidden layers  

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern using -1/1 for tanh

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad
    
    if step % 20 == 0:
        print(f"step {step:3d} loss = {loss.data:.4f}")

print("\nPredictions after training")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")

def gradient_checking(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    print(y)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val -h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff

# # testing it on some complex function

def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_checking(expr, 0.5)
print(f"Autodiff: {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")  # should be less 1e-5


x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()
# # this backward step will call build_topo(y). It walks backward from there.
# # topo = [Value(1.0), x1, x2, a, b, y]
# # to do backpropagation python reverse this
# #[y, b, a, x2, x1, Value(1.0)] 
# # seed gradient will be iniitialized y.grad = 1.0
# # now python steps through the reversed(topo) loop and triggers the unique inner _backward() 
# # functions that were frozen inside each object during the forward pass
# # executing y._backward() the relu gate: the object y came from relu operatoin on b. and it
# # will run the stored recipe  for ReLU. it will give b.grad = 1.0
# # 2. b._backward(addtion gate): object b came from a and adding Value(1.0). the stored 
# # recipe for __add__() will run addition operation acts like gradient distributor and the 
# # resulting gradient will be a.grad = 1.0 which was also the previous gradient.
# # 3. a._backward()(The multipicative Gate). object a came from multiplying x1 and x2. so 
# # the recipe for mul function will run self being x1 and other being x2 and out being a.
# # x1.grad = 3.0 * 1.0 = 3.0
# # x2.grad = 2.0 * 1.0 = 2.0

print(f"y = {y.data}") 
print(f"dy/dx1 = {x1.grad}")
print(f"dy/dx2 = {x2.grad}") 

# verify against torch module
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()
print(f"Pytorch dy/dx1 = {x1.grad.item()}")
print(f"Pytorch dy/dx2 = {x2.grad.item()}")


##-------------------------------------EXERCISES--------------------------------------
print("-"*20 + "Exercise 1" + "-"*20)

# exercise 1: Add __pow__ to the Value class so you can compute x ** n. Verify that d/dx(x^3)
#  at x=2 equals 12.0.


x = Value(2.0)
y = x ** 3
y.backward()
print(f" the gradient at x = 2 : {x.grad}")

print(y.data)

# Exercise 2: Add tanh as an activation function. Verify that tanh'(0) = 1 and tanh'(2) = 0.0707 (approx).

x1 = Value(0.0)
x2 = Value(2.0)


y1 = x1.tanh()
y1.backward()
y2 = x2.tanh()
y2.backward()

print(f"tan'(0) is {x1.grad:.4f}")
print(f"tan'(2) is : {x2.grad:.4f}")

# exercise 3: Build a computation graph for a single neuron: y = relu(w1x1 + w2x2 + b).
# Compute all five gradients and verify against PyTorch.
## i'm using my Neuron class first then i will do with value class it self and then we will tally it against pytorch.

x_input = [Value(1.1), Value(1.9)]
com_graph = Neuron(n_inputs=2)

com_graph.w[0].data = 5.0 # so that it can be easily worked out with in pytorch
com_graph.w[1].data = 2.4 # pytorch easiness
com_graph.b.data = 0.5

y_out = com_graph(x_input)
y_out.backward()

print("--computation graph -----------------outputs")
print(f"w1 grad: {com_graph.w[0].grad}")
print(f"w2 grad: {com_graph.w[1].grad}")
print(f"x1 grad: {x_input[0].grad}")
print(f"x2 grad: {x_input[1].grad}")
print(f"bias b grad: {com_graph.b.grad}")

print("-"* 50)
# using Value class to make same stuff.

x1 = Value(1.1)
x2 = Value(1.9)
w1 = Value(5.0)
w2 = Value(2.4)  
b  = Value(0.5)    

y_output = w1*x1 + w2*x2 + b
final_output = y_output.relu()
final_output.backward()

print(f"w1 grad: {w1.grad}")
print(f"w2 grad: {w2.grad}")
print(f"x1 grad: {x1.grad}")
print(f"x2 grad: {x2.grad}")
print(f"bias b grad: {b.grad}")

print("-"*25 + "tensor verification" + "-"*25)

x1 = torch.tensor(1.1, requires_grad=True)
x2 = torch.tensor(1.9, requires_grad=True)
w1 = torch.tensor(5.0, requires_grad=True)
w2 = torch.tensor(2.4, requires_grad=True)  
b  = torch.tensor(0.5, requires_grad=True)

y_output = w1*x1 + w2*x2 + b

y_final = y_output.relu()
y_final.backward()

print(f"w1 grad: {w1.grad.item():.2f}")
print(f"w2 grad: {w2.grad.item():.2f}")
print(f"x1 grad: {x1.grad.item():.2f}")
print(f"x2 grad: {x2.grad.item():.2f}")
print(f"bias b grad: {b.grad.item():.2f}")

## exericise 4: Implement forward-mode autodiff using dual numbers. 
# Create a Dual class and verify it gives the same derivatives as your reverse-mode engine.

class Dual:
    def __init__(self, value, d = 0.0):
        '''
        This works on the formula a + b*epsilon, where epsilon^2 = 0
        '''
        self.value = value
        self.d = d   # the dual part derivative

    def __repr__(self):
        return f"Dual(value={self.value:.4f}, d={self.d:.4f})"
    
    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)

        addition = self.value + other.value
        derivative = self.d + other.d
        return Dual(addition, derivative)
    
    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)

        multiplication = self.value * other.value
        derivative_mul = self.d * other.value + self.value * other.d 
        return Dual(multiplication, derivative_mul)
    
    def relu(self):
        addition = max(0, self.value)

        # forward pass derivative
        derivative = self.d if self.value > 0.0 else 0.0
        return Dual(addition, derivative)

        
# testing with our already defined single neuron equation of last exercies(w1*x1 + w2*x2 + b)

x1 = Dual(1.1, d=0.0)
x2 = Dual(1.9, d=0.0)
w1 = Dual(5.0, d=1.0)  # seed: we want it's derivative
w2 = Dual(2.4, d=0.0)
b = Dual(0.5, d=0.0)

# running the forward pass
y = (w1 * x1) + (w2 * x2) + b
y_final = y.relu()

# reading the answer: no backward pass is required
print(f"Final value: {y_final.value:.4f}")
print(f"derivative dy/dw1: {y_final.d}")
