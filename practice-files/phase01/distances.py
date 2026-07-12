'''
Here i will make various distances from scratch and apply them in the problem they are most suited to.
'''
import math
import random

# let' build.

def l_1_norm(x):
    return sum(abs(xi) for xi in x)

def l_2_norm(x):
    return math.sqrt(sum(xi**2 for xi in x))

def lp_norm(x, p):
    if p == float('inf'):
        return max(abs(xi) for xi in x)
    return sum(abs(xi) ** p for xi in x) ** (1/p)

def linf_norm(x):
    return max(abs(xi) for xi in x)

def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def l2_distance(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

def lp_distance(a, b, p):
    diff = [ai - bi for ai, bi in zip(a, b)]
    return lp_norm(diff, p)

def linf_distance(a, b):
    return max(abs(ai - bi) for ai, bi in zip(a, b))

def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def cosine_similarity(a, b):
    dot = dot_product(a, b)
    norm_a = l_2_norm(a) 
    norm_b = l_2_norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot/ (norm_a * norm_b)

def cosine_distance(a, b, **kwargs):
    cos_sim = cosine_similarity(a, b)
    return 1.0 - cos_sim

def mahalanobis_distance(x, y, cov_mat):
    n = len(x)
    diff = [xi - yi for xi, yi in zip(x, y)]
    inv_cov = invert_matrix(cov_mat)

    temp = [0.0] * n
    for i in range(n):
        for j in range(n):
            temp[i] += diff[j] * inv_cov[i][j]
    result = sum(temp[i] * diff[i] for i in range(n))
    return math.sqrt(max(0, result))

# def invert_matrix(matrix):
def invert_matrix(matrix):
    n = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                max_row = row
        augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or near-singular")
        for j in range(2 * n):
            augmented[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[col][j]

    return [row[n:] for row in augmented]

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

def jacard_distance(set_a, set_b):
    return 1.0 - jaccard_similarity(set_a, set_b)

def wasserstein_1d(p, q):
    assert len(p) == len(q), "Distributions must have the same number of bins"
    n = len(p)
    cdf_p = [0.0] * n
    cdf_q = [0.0] * n

    cdf_p[0] = p[0]
    cdf_q[0] = q[0]
    for i in range(1, n):
        cdf_p[i] = cdf_p[i - 1] + p[i]
        cdf_q[i] = cdf_q[i - 1] + q[i]

    return sum(abs(cdf_p[i] - cdf_q[i]) for i in range(n))

def find_nearest_neighbor(query, dataset, distance_fn, **kwargs):
    best_idx = 0
    best_dist = float('inf')
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist

def find_k_nearest(query, dataset, distance_fn, k=5, **kwargs):
    distances = []
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return distances[:k]


## Exercise1: Compute L1, L2, and L-infinity distances between (1, 2, 3) and (4, 0, 6). Verify 
# that L-inf <= L2 <= L1 always holds for any pair of points. Prove why this ordering is guaranteed.

a = (1, 2, 3)
b = (4, 0, 6)
linf = linf_distance(a, b)
l1 = l1_distance(a, b)
l2 = l2_distance(a, b)

print(f"the linf distance:{linf}, l1 distance:{l1} and l2 distance:{l2:.4f} between points {a} and {b}")
# linf is smallest(3) because it only gives distance along the biggest dimentions, so only the biggest axis distance will be considered
# followed by linf is l2 distance(4.69) which comes second here because it is a straight line difference between points and it ignores
# the grids system. The last l1 distance is largest(8) because it follows the grids and summed overal distance of moving along each-axis
# to reach a point.

## Exercise 2: Create two vectors where cosine similarity is high (> 0.9) but L2 distance is large (> 10). Explain geometrically 
# what is happening. Then create two vectors where cosine similarity is low (< 0.3) but L2 distance is small (< 0.5).

a = (1.0, 2.0)
b = (10.0 , 15.0)
cosine_sim = cosine_similarity(a, b)  # here cosine similarity is high 
l2_dist = l2_distance(a, b)  # l2 is high 
print(l2_dist, cosine_sim)

# cosine similarity low and l2_small
a = (-0.9, 2.0)
b = (1.0, 1.0)
cos_sim_low = cosine_similarity(a, b)
l2_dist_low = l2_distance(a, b)
print(l2_dist_low, cos_sim_low)

# here i can't make the distance l2 less than 0.5 and and keep the cosine similarity also low(0.3) because for distance to be 
# low the cosine similarity will be high as the vectors will be in same direction and closer to each other.. the relationshitp is
# kind of inverse between those 2.

## Exercise 3: Implement a function that takes a dataset and a query point and returns the nearest neighbor 
# under L1, L2, cosine, and Mahalanobis distance. Find a dataset where all four disagree on which point is nearest.

def calculate_covariance_matrix(data):
    # data is a list of lists (N x M)
    n_samples = len(data)
    n_features = len(data[0])
    means = [sum(row[j] for row in data) / n_samples for j in range(n_features)]
    
    cov_mat = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            variance_sum = sum((row[i] - means[i]) * (row[j] - means[j]) for row in data)
            cov_mat[i][j] = variance_sum / (n_samples - 1)
    return cov_mat

def find_dist(query, dataset, distance_fn, **kwargs):
    best_idx = 0
    best_dist = float('inf')
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist

## testing dataset:
dataset = [
    [1.9, 5.0],  # Point 0: Cosine Target (Closest angle)
    [2.9, 11.1],  # Point 1: L1 Target (Manhattan taxicab step balance)
    [1.0, 0.45],  # Point 2: L2 Target (Closest Euclidean distance)
    [11.0, 0.2]   # Point 3: Mahalanobis Target (High variance axis)
]
query = [0.0, 1.0]

cov = calculate_covariance_matrix(dataset)

metrics = {
    "L1 (Manhattan)": lambda q, p, **kw: l1_distance(q, p),
    "L2 (Euclidean)": lambda q, p, **kw: l2_distance(q, p),
    "Cosine": cosine_distance,
    "Mahalanobis": mahalanobis_distance
}

# Running the test.
print(f"Query Point: {query}\n" + "-"*50)

for name, fn in metrics.items():
    idx, dist = find_dist(query, dataset, fn, cov_mat=cov)
    print(f"{name:15} -> Nearest Neighbor Index: {idx} (Point: {dataset[idx]}, Dist: {dist:.4f})")

## Exercise 4: Compute the Wasserstein distance between [0.5, 0.5, 0, 0] and [0, 0, 0.5, 0.5] by hand using the CDF method. 
# Then compute it between [0.25, 0.25, 0.25, 0.25] and [0, 0, 0.5, 0.5]. Which is larger and why?

p = [0.5, 0.5, 0, 0]
q = [0, 0, 0.5, 0.5]
wass_dist = wasserstein_1d(p, q)  # this one is larger 2.
wass_dist_n = wasserstein_1d([0.25, 0.25, 0.25, 0.25], [0, 0, 0.5, 0.5])  # this is smaller 1.
print(f"wass of p and q is {wass_dist} and a and b is {wass_dist_n}")

# it was 2.0 for the distribution p to move to q because we have to move points(dirt) from bins on the left to all the way to 
# right? so 100% of mass has to travel in this one, on another set of point the dust is already distributed kind of uniformally,
# so to move it over to the points next is just from position/bins (0, 1) and needed to move to right side to position 2, 3.
