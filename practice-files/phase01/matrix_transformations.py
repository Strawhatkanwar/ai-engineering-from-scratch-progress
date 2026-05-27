import math
import numpy as np

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a , cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    # print(cols_a, cols_b)
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(cols_a)) 
                for j in range(cols_b)
        ]
        for i in range(rows_a)
    ]
point = [1.0, 0.0]
angle = math.pi / 4 

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")


####====================exerise 1----------------------------------
## Apply rotation, scaling, and shearing to a 
# unit square (corners at [0,0], [1,0], [1,1], [0,1]). Print the transformed corners 
# for each. Verify that rotation preserves distances between corners.

#solution, so we have set of vectors([0,0],[0,1],[1,1],[1,0])
square = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]

def dist(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + 
        (p1[1] - p2[1]) ** 2 
    )
rotation = {}
for point in square:
    rotated = mat_vec_mul(rotation_2d(angle), point)
    sheared = mat_vec_mul(shearing_2d(0, 1), point)
    scaled = mat_vec_mul(scaling_2d(2, 3), point)


    print(f"\npoint: {point}")
    print(f"Rotated: {rotated}")
    print(f"Scaled: {scaled}")
    print(f"sheard: {sheared}")

print(f"rotation dict is: {rotation}")
r1 = mat_vec_mul(rotation_2d(angle), square[0])
r2 = mat_vec_mul(rotation_2d(angle), square[1])

print(r1, r2)
d_before = dist(square[0], square[1])
d_after = dist(r1, r2)
print(f"distance after the trnasformation: {d_after}")

# exercise 2-------------------------------------------
# Find the eigenvalues of the matrix [[4, 2], [1, 3]] by hand using the 
# characteristic equation. Then verify with your from-scratch function and with NumPy.


def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[4, 2], [1, 3]]
eigen_vals = eigenvalues_2x2(A)
print(eigen_vals)
eigenvec_1 = eigenvector_2x2(A, 5)
eigenvec_2 = eigenvector_2x2(A, 2)
print(eigenvec_2, eigenvec_1)

# verifying with nump
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\n Eigenvalues : {eigenvalues}")
print(f"\n Eigenvectors (columns): \n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f" A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

## exercise 3: Create a composition of three transformations (rotate 30 degrees, 
# scale by [1.5, 0.8], shear with kx=0.3) and apply it to 8 points arranged in a circle. 
# Print before and after coordinates. Compute the determinant of the composed matrix and 
# verify it equals the product of the individual determinants.

# creating composition using our function defined above?
print("#"* 30)
angle = math.pi / 6
rotate_30 = rotation_2d(angle)
scaled = scaling_2d(1.5, 0.8)
sheared = shearing_2d(0.3, 0)


points = [
    [0, 3],
    [2, 2],
    [3, 0],
    [2, -2],
    [0, -3],
    [-2, -2],
    [-3, 0],
    [-2, 2]
]

# order of our transformations are HSR(right to left, rotate, scale, shear) so (HSR)v
mat_mul_first = mat_mul(scaled, rotate_30)
mat_mul_after = mat_mul(sheared, mat_mul_first)
composed_mat = mat_mul_after
print(composed_mat)
# computing the determinant
det_comp = np.linalg.det(np.array(composed_mat))
print(det_comp)
det_r = np.linalg.det(np.array(rotate_30))
det_sc = np.linalg.det(np.array(scaled))
det_sh = np.linalg.det(np.array(sheared))



print(f"checking if det equality: {det_comp == det_r * det_sc * det_sh}")

for i in points:
    print(f"points before composition: {i}")
    transformed = mat_vec_mul(mat_mul_after, i)
    print(f"After apply all transformation i.e. rotate, scale, shear in orderthe resulting points transformed vector for point {i} is {transformed}")
