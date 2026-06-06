import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None
        # self.eigenvector = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        # self.eigenvector = eigenvectors
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T
    

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced):
        # X_reduced shape: (n_samples, n_components)
        X_reconstruced = X_reduced @ self.components  #(n_samples, reduced_dim) @ ()
        X_reconstruced += self.mean  # adding mean back to re-center the data.
        return X_reconstruced

## testing our function on synthetic data
np.random.seed(42)
n_samples = 500

t = np.random.uniform(0, 2 * np.pi, n_samples)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

X_synthetic = np.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synthetic)

print(f"Original shape: {X_synthetic.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")


## Mnist in 2d

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=500)
X_pca50 = pca_mnist.fit_transform(X_mnist)
print(f"50 components capture {sum(pca_mnist.explained_variance_ratio_)} of variance")
print(f"50 components capture {sum(pca_mnist.explained_variance_ratio_)} of variance")

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist)
print(f"2 components capture {sum(pca_2d.explained_variance_ratio_)} of variance")

# comparing with sklearn
sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

print(f"\nOur PCA explained variance:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
print(f"Max absolute difference: {diff.max():.10f}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist)
print(f"\nt-SNE output shape: {X_tsne.shape}")

## UMAP Comparison

try:
    from umap import UMAP
    
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_mnist)
    print(f"UMAP output shape: {X_umap.shape}")
except ImportError:
    print("Install umap-learn: pip install umap-learn")


# using PCA as preprocessing before a classifier.

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist, y_mnist, test_size=0.2, random_state=42
)
print(X_train.shape, y_train.shape)
results = {}

for k in [10, 30, 50, 100, 200]:
    pca_k = SklearnPCA(n_components=k)
    X_tr = pca_k.fit_transform(X_train)
    X_te = pca_k.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    var_captured = sum(pca_k.explained_variance_ratio_)
    results[k] = (acc, var_captured)
    print(f"k={k:>3d} accuracy={acc:.4f} variance={var_captured:.4f}")


print("-" * 25 + "EXERCISES" + "-"*25)

## Exercise 1: Modify the PCA class to support inverse_transform. Reconstruct MNIST digits from 10, 50, and 200 components. 
# Print the reconstruction error (mean squared difference from the original) for each.

components = [10, 50, 200]

for n_components in components:
    pca_mnist = PCA(n_components)
    mnist_reduced = pca_mnist.fit_transform(X_mnist)
    print(mnist_reduced.shape)
    # finding the reconstructoin error:
    # projecting data to k-dimention
    mnist_reconstructed = pca_mnist.inverse_transform(mnist_reduced)
    print(mnist_reconstructed.shape)
    # find out the way to equate matrices
    mse = np.mean((X_mnist - mnist_reconstructed)**2)
    print(f"Components: {n_components:>3d} | Reconstruction MSE: {mse:.4f}")

##Exercise 2: Run t-SNE on the same MNIST subset with perplexity values of 5, 30, and 100. 
# Describe how the output changes. Why does perplexity affect cluster tightness?

perp_values = [5, 30, 100]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, perp in enumerate(perp_values):
    tsne_mnist = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne_mnist.fit_transform(X_mnist)

    # plotting the 2dProjection and color points by their actual MNIST digit label
    ax = axes[i]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_mnist, cmap='tab10', s=1, alpha=0.6)
    ax.set_title(f"Perplixity = {perp}")
    ax.axis("off")
    #print(f"The shape of transformed mnist is {X_tsne.shape}")
plt.colorbar(scatter, ax=axes[-1], label="Digital Label")
output_filename = "tsne_perplexity_comparison.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved to {output_filename}!")

# I can see with perplexity 5, the clusters are separated clearly but they have more space
# btween them and it's hard to tell it it belong to certain cluster without color coding. so 
# no clear seprated clusteres.
# with perplexity 30, i can see the real cluster formation even without color coding we can
# tell that this cluster is different from other as they are visibly have distance between them
# and the point belonging to a cluster are tightly clustered/packed together. so that good.
# with preplixity 100, i see the cluster are more condensed in their clusters but i see more of
# a mix between points of other clusters and overall it's not bad either..

# winner is preplexity 30. Tsne tries to fit guassian distribution to each point in 784d space
# and tries to club together the point near that space, the perplexity paramters here determines 
# the variance for that gaussians, and tsne adjusts the variance dynamically to each individual
# point so that it maches our perplexity value. Also perplexity is 2 to power of entropy of distribution
# so when it's small entropy is less and less points will be in that gaussian. 
# but if it's moderate(like 30-50) the points that really are near and form a cluster would be clubbed in,
# and if it's too much it might club the points that are even far in some dimention to its own
# guassian.


## Exercise 3: Take a dataset with 50 features where only 5 are informative (generate one with sklearn.datasets.make_classification). 
# Apply PCA and check whether the explained variance curve correctly identifies that the data is effectively 5-dimensional.

n_samples = 1000
X, y = make_classification(n_samples, n_features=50, n_informative=5)
print(X.shape, y.shape)

pca_sklearn = SklearnPCA()
X_reduced = pca_sklearn.fit_transform(X)

print(X_reduced.shape)
print(pca_sklearn.explained_variance_ratio_[:10])

## Pca doesn't seem to capture strictly all informative variance in 5 dimentions the first 2 captures descent amount [0.133, 0.116] 
# after that there's shape decline in 3rd(0.0431) and steady for 4th(0.025), 5th(0.0225) and 6th(0.02202) and so on. it means it failts to
# capture the non-linearity of features as it works great on linear relationships. Also make_classification generated data with
# non-linear relationships and multiple clusters per class. Since PCA is strictly a linear dimentionality reduction technique it 
# fails to capture this non-linearity, causing the informative signals to bleed across many extra noise dimentions.