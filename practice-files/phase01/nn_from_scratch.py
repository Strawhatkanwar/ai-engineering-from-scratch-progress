'''
Here we will do Neural network intuition from scratch and build our own Vector and Matrix
Classes and with element-wise operations, matrix multiplication, transpose, determinant,
and inverse,
- Implement single dense neural network layer using our scratch matrix class only.
- Explain the broadcasting rules and how bias addition works in NN frameworks.
'''
import random
import numpy as np

class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"
    
    def __add__(self, other):
        return Vector([a + b] for a, b in zip(self.data, other.data))
    
    def __sub__(self, other):
        return Vector([a - b] for a, b in zip(self.data, other.data))
    
    def __mul__(self, scaler):
        return Vector([x * scaler] for x in self.data)

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
    

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix ({self.shape}):\n {rows_str}"
    
    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])
    
    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])
    
    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det
    
    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])
    
    def cofactor_matrix(self):
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                # minor matrix with row i and col j removed
                minor = Matrix([
                    [self.data[r][c] for c in range(self.cols) if c != j]
                    for r in range(self.rows) if r != i
                ])
                cofactor = ((-1) ** (i + j)) * minor.determinant()
                row.append(cofactor)
            result.append(row)
        return Matrix(result)
    
    def inverse_3x3(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        cofactors = self.cofactor_matrix()
        adjugate = cofactors.transpose()
        return adjugate.scalar_multiply(1/det)
    
    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
    
# see it in action

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
C = Matrix([[3, 5], [2, 1]])
print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B))
print("A^T =", A.transpose().data)
print("Det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
# print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)



# step of connecting it all to Nueral networks

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
# print(pre_activation)
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")

print(I.data)
# exercise 1:Verify the inverse. Multiply A @ A.inverse_2x2() and confirm you get the identity matrix. \
# Try it with three different 2x2 matrices. What happens when the determinant is zero?

## helper function
def approximate_equal(m1, m2, tol=1e10):
    for i in range(m1.rows):
        for j in range(m1.cols):
            if abs(m1.data[i][j] - m2.data[i][j]) > tol:
                return False
    return True

print("A @ A^-1:", A.matmul(A.inverse_2x2()).data == I.data)
print("B @ B^-1:", B.matmul(B.inverse_2x2()).data == I.data)
print("C @ C^-1: ", approximate_equal(C.matmul(C.inverse_2x2()), I))


X = Matrix([[1, 2], [2, 4]]) # this is linearly dependent matrix. as first vector is linear combination of 2nd and vice versa.
print(X.determinant())
# X.inverse_2x2()  # it will yield in value error(commenting it to run full code)


### exercise 2: Implement 3x3 inverse. Extend the Matrix class to compute inverses for 3x3 matrices using the adjugate method. 
# Test it against NumPy's np.linalg.inv.
# made the cofactors_matrix method in Matrix class and also added inverse_3x3() method as well.

A = Matrix([[1, 2, 3], [3, 1, 7], [9, 2, 1]])
A_num = np.array([[1, 2, 3], [3, 1, 7], [9, 2, 1]])
print("A @ A^-1:", A.inverse_3x3())

num_out = np.linalg.inv(A_num)
print(num_out)

### Exercise 3: Build a two-layer network. Using only your Matrix class (no NumPy), create a two-layer neural network: 
# input (3) -> hidden (4) -> output (2). Initialize random weights, run a forward pass, and verify all shapes are correct.

input = Matrix([[1.5], [2.1], [5.2]])  # shape (3, 1) 3 features

weights = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(4)] # shape (4, 3)
weights = Matrix(weights)

bias = Matrix([[0.3], [0.2], [.18], [.09]])  # matrix of shape (4, 1)
pre_activation = weights.matmul(input) + bias  # forward pass
output = relu_matrix(pre_activation)  # output from first layer (4, 1)

# 2nd layer:
weights2 = Matrix([[random.uniform(-1, 1) for _ in range(4)] for _ in range(2)])
bias2 = Matrix([[0.11], [0.22]])  # shape (2, 1 because final output is 2, 1)
final_output = weights2.matmul(output) + bias2


print(f"input shape: {input.shape}")
print(f"weights shape: {weights.shape}")
print(f"output shape: {output.shape}")
print(f" Output: {output.data}")
print(f"the final output is :{final_output}")
print(f"the final output shape is :{final_output.shape}")

