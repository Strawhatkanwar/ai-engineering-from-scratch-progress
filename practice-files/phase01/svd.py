'''
here i made svd from scratch and compare it against svd fromm numpy, calculate error of reconstructed matrices, for the 
document term matrix from scratch for 100 docs and 50 columns(words) for 3 topics and found out that first 3 singular values were
highest followed by sharp decrease. I did all exercises in this python file(svd.py) 
'''

import numpy as np

def power_iteration(M, num_iter=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v  / np.linalg.norm(v)

    for _ in range(num_iter):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)
    
    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iter=200)

        if eigenvalue < 1e-10:
            break
        
        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V

# testing with numpy

np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("Numpy singular values: ", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T

print(f"Recondstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")

### Image compression

def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d} error={error:.4f} storage={ratio:.1%}")

# noise reduction

np.random.seed(42)

clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))

noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)

denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error: {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement: {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")

## Pseudoinverse

A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution: {x_svd}")
print(f"np.linalg.lstsq solution: {x_lstsq}")
print(f"np.linalg.pinv solution: {x_pinv}")


print("-"*25 + "EXERCISES" + "-"*25)

## Exercise 1: Implement the full SVD from scratch without using power iteration. Instead, compute the 
# eigendecomposition of A^T A to get V and the singular values, then compute U = A V Sigma^{-1}. Compare numerical 
# accuracy with your power iteration version and with NumPy.    

def svd_by_eig(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)
    AtA = A.T @ A
    eigenvalues, V = np.linalg.eigh(AtA) 
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    V = V[:, sorted_idx]

    # slice to keep top k
    eigenvalues = eigenvalues[:k]
    V = V[:, :k]

    # calculating sigma
    sigmas = np.sqrt(np.maximum(eigenvalues, 0))

    # compute  U = A @ V @ Sigma^-1
    AV = A @ V
    U =  AV / sigmas
    return U, sigmas, V

# check of matrix A
A = np.random.randn(6, 5)

U_ours, S_ours, V_ours = svd_by_eig(A)
U_pi, S_pi, V_pi = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
k_val = min(A.shape)
print(f"SVD without using power iteration is: {np.round(S_ours, 6)}")
print(f"SVD using power iteration is : {np.round(S_pi, 6)}")
print(f"SVD using numpy is {np.round(S_np[:k_val], 6)}")

# comparing the matrix outut
diff_u = np.abs(np.abs(U_ours) - np.abs(U_pi)).max()
diff_s = np.abs(S_ours - S_pi).max()
diff_v = np.abs(np.abs(V_ours) - np.abs(V_pi)).max()

print(f"Max absolute difference in U: {diff_u:.10f}")
print(f"Max absolute difference in S: {diff_s:.10f}")
print(f"Max absolute difference in V: {diff_v:.10f}")


## Exercise 2: Load a real grayscale image (or convert one to grayscale). Compress it at ranks 1, 5, 10, 25, 50, 100. For each rank, 
# compute the compression ratio and the relative error. Find the rank where the image becomes visually acceptable.
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("sushi.jpg")
img = img.convert('L')
image = np.asarray(img)
m, n = image.shape
print(image.shape)

ranks = [1, 5, 10, 25, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, k in enumerate(ranks):
    compressed = compress_image_svd(image, k)

    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = m * n
    compressed_size = k * (m + n + 1)
    ratio = compressed_size / original_size
    print(f"for K={k:3d} the relative error {error:.4f} and Storage ratio is {ratio:2%}")

    # plotting the results
    ax = axes[i]
    ax.imshow(compressed, cmap='gray')
    ax.set_title(f"Rank K = {k}\nError: {error:.2f} | Storage: {ratio:.1%}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sushi_svd_compression.png", dpi=300)
print("Visual comparison saved to sushi_svd_compression.png")

# looking at the produced images the one with rank 50 is acceptable image i can see all the features of image and with rank 100
# i can say it looks like original..

## Exercise 3: Create a 100x50 document-term matrix with 3 synthetic topics. Each topic has 5 associated terms. Add noise. 
# Apply SVD and verify that the top 3 singular values are much larger than the rest. 
# Project documents into the 3D latent space and check that documents from the same topic cluster together.
n_docs = 100
n_terms = 50
n_topics = 3
base_mat = np.zeros((n_docs, n_terms))

# first 33 will have topic 1(animals), other 33% another topic(birds) and last topic3(trees)
doc_labels = np.array([0]*33 + [1]*33 + [2]*34)

for doc_index in range(n_docs):
    topic = doc_labels[doc_index]

    # first 5 cols will have 5 words associated with topic1 and another five with topic 2 and other 5 with topic 3 after that
    # there will be noise.
    start_col = topic * 5
    end_col = start_col + 5
    base_mat[doc_index, start_col:end_col] = np.random.uniform(5.0, 10.0, size=5)

    base_mat[doc_index, :] += np.random.uniform(0.0, 2.0, size=n_terms)

# adding noise
noise = np.random.normal(0, 1.5, size=(n_docs, n_terms))
base_mat_noisy = base_mat + noise

base_mat_noisy = np.maximum(base_mat_noisy, 0)

U, S, Vt = np.linalg.svd(base_mat_noisy, full_matrices=False)

print("Top 10 singular Values:")
print(np.round(S[:10], 2))

# project document into 3d latent space
X_3d = U[:, :3] * S[:3]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']

topics_names = ['animals', 'birds', 'trees']
for topic_id in range(n_topics):
    mask = (doc_labels == topic_id)
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
               c=colors[topic_id], label=topics_names[topic_id], s=40)

ax.set_title("Documents Projected into 3D Latent Space")
ax.set_xlabel("Latent Topic 1")
ax.set_ylabel("Latent Topic 2")
ax.set_zlabel("Latent Topic 3")
ax.legend()

plt.savefig("latent_topic_clusters.png", dpi=300)
print("\nPlot saved to latent_topic_clusters.png")


## Exercise4: Generate a clean low-rank matrix (rank 3, size 50x40) and add Gaussian noise at different levels 
# (sigma = 0.1, 0.5, 1.0, 2.0). For each noise level, find the optimal truncation rank by sweeping k from 
# 1 to 40 and measuring reconstruction error against the clean matrix. Plot how the optimal k changes with noise level.
n_cols = 40
n_rows = 50
rank = 3
base_mat = np.random.randn(50, 40)

a = np.random.randn(n_rows, rank)
b = np.random.randn(rank, n_cols)

new_mat = a @ b # it will be rank 3 matrix with size 50 x 40

sigmas = [0.1, 0.5, 1.0, 2.0]
opt_ks = []

plotting_dict = {}

for sigma in sigmas:
    noise =  np.random.normal(0, sigma, size=(n_rows, n_cols))
    noisy_mat = new_mat + noise

    best_k = 1
    lowest_error = float('inf')
    errors_for_this_sigma = []

    for k in range(1, 41):
        compressed = compress_image_svd(noisy_mat, k=k)

        error = np.linalg.norm(new_mat - compressed) / np.linalg.norm(new_mat)
        errors_for_this_sigma.append(error)
        # tracking which k gives the absolute lowest error against true signal
        if error < lowest_error:
            lowest_error = error
            best_k = k
        
    opt_ks.append(best_k)
    plotting_dict[sigma] = errors_for_this_sigma
    print(f"Noise Sigma = {sigma:.<3} | Optimal Truncation Rank K = {best_k} (Lowest error: {lowest_error:.4f})")

# plotting
plt.figure(figsize=(8, 5))
for sigma in sigmas:
    # Pull the 40 error values we saved for this specific sigma
    errors = plotting_dict[sigma]
    
    # Plot the curve
    plt.plot(range(1, 41), errors, label=f"Noise Sigma = {sigma}")

# Add a vertical dashed line at k=3 to show our true rank
plt.axvline(x=3, color='black', linestyle='--', alpha=0.7, label='True Rank (k=3)')

plt.title("Reconstruction Error vs. Truncation Rank (k)")
plt.xlabel("Truncation Rank (k)")
plt.ylabel("Relative Error against CLEAN Matrix")
plt.legend()
plt.grid(True, linestyle='--')

plt.savefig("error_curves_complete.png", dpi=300)
print("Complete analysis plot saved as error_curves_complete.png")

# relative error against clean matrix(new_mat) drops down to minimum at k at 3 no matter how much noise we add to it
# sigma(0.1 , 0.5, 1.0, 2.0). after k=3 error starts going back up because we're measuring it against the clean matrix. so 
# there's nothing to reconstruct other than random gaussian noise after k =3.



