import random
import math
import numpy as np

class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)
    
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])
    
    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])
    
    def dot(self, other):
        return sum([a * b for a, b in zip(self.components, other.components)])
    
    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5
    
    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])
    
    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())
    
    def angle_between(self, other):
        mag = self.magnitude() * other.magnitude()
        dot = self.dot(other)
        if mag == 0:
            return 0
        # calculating the cos theta
        cos_theta = (dot/mag)
        radians = math.acos(cos_theta)
        return math.degrees(radians)



    def __repr__(self):
        return f"Vector({self.components})"
    
a = Vector([1, 0, 0])
b = Vector([0, 1, 0])

print(f"the angle between vec a and vec b is: {a.angle_between(b)}")
print(f"the other way around : {b.angle_between(a)}")

class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][j] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)
    
    def transpose(self):
        return Matrix([
            [self.rows[i][j] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])
    
    def __repr__(self):
        return f"Matrix({self.rows})"

rotation_90  = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90: {rotated}")   

# exammple using real system of ML


random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication. ")

# ------------------------------LI and projection-----------------------------
## linear independence and projection

def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col] > 1e-10):
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col] 
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n

def project(a, b):
    scaler = a.dot(b) / b.dot(b)
    return Vector([scaler * x for x in b.components])

def gram_schimdt(vectors):
    orthonormal = []
    for v in vectors:
        w = v 
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal

v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])

# basis = gram_schimdt([v1, v2, v3])
# for i, u in enumerate(basis):
#     print(f"u{i+1} = {u}")
#     print(f" |u{i+1}| = {u.magnitude():.6f}")

# print(f"u1 . u2 = {basis[0].dot(basis[1]):.6f}")
# print(f"u1 . u3 = {basis[0].dot(basis[2]):.6f}")
# print(f"u2 . u3 = {basis[1].dot(basis[2]):.6f}")

# -------------------exercise 2--------------------------------
scal_mat = np.matrix([[2,0], [0,3]])
print(scal_mat)
array = np.array([1, 1])
print(scal_mat @ array)


#---------------exerices 3-------------------------------------
vec_1 = np.array([np.random.randint(1, 1000) for i in range(50)])
vec_2 = np.array([np.random.randint(1, 1000) for i in range(50)])
vec_3 = np.array([np.random.randint(1, 1000) for i in range(50)])
vec_4 = np.array([np.random.randint(1, 1000) for i in range(50)])
vec_5 = np.array([np.random.randint(1, 1000) for i in range(50)])

v1 = Vector(vec_1)
v2 = Vector(vec_2)
v3 = Vector(vec_3)
v4 = Vector(vec_4)
v5 = Vector(vec_5)

cosine = v1.cosine_similarity(v2)
print(cosine)

# calculating the score pairwise annd then check which pair is closes,
track = []
# score = 0
pairs = [(v1, v2), (v1, v3), (v1, v4), (v1, v5), (v2, v3), (v2, v4), (v2, v5),
        (v3, v4), (v3, v5), (v4, v5)]
names = [("v1", "v2"), ("v1", "v3"), ("v1", "v4"), ("v1", "v5"), ("v2", "v3"),
         ("v2", "v4"), ("v2", "v5"), ("v3", "v4"), ("v3", "v5"), ("v4", "v5")]

for (a, b), name in zip(pairs, names):
    score = a.cosine_similarity(b)
    track.append((name, score))

best = max(track, key=lambda x: x[1])
# print(track)
print(f"best similar {best[0]} with score {best[1]:.4f} ")


# ---------------exercise 4--------------------------------------------------
basis = gram_schimdt([v1, v2, v3])

#print(f"basis for gram_schmidt is : {basis}")


# magnitude 1
for i, u in enumerate(basis):
    print(f"|u{i+1}| = {u.magnitude():.6f}")
    # print(i, u)
# checking dot products equals 0
for i in range(len(basis)):
    for j in range(i+1, len(basis)):
        dot = basis[i].dot(basis[j])
        print(f"u{i+1} . u{j+1} = {dot:.6f}")

##----------------exercise 5-----------------rank of a 3x3 matrix of 2------------

mat_a = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
])
print(np.linalg.matrix_rank(mat_a))

vector1 = Vector([1, 2, 3])
vector2 = Vector([1, 1, 1])

print(f"the projection is : {project(vector1, vector2)}")