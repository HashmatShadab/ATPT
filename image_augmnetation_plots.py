from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Resample for completeness
np.random.seed(42)
w_samples = np.array([np.random.dirichlet([1.0, 1.0, 1.0]) for _ in range(10000)])
m_samples = np.array([np.random.beta(1.0, 1.0) for _ in range(10000)])

"""
This script visualizes the properties of random weights ("w") and mixing coefficient ("m")
used in the AugMix data augmentation strategy.

Specifically:
- "w" is sampled from a Dirichlet(1,1,1) distribution, producing 3 non-negative values that sum exactly to 1.
- "m" is sampled from a Beta(1,1) distribution, which is equivalent to a uniform distribution on [0,1].

The plots generated are:
1. Histogram of the sums of w to verify they sum to 1.
2. 3D scatter plot showing (w[0], w[1], w[2]) points lying on the 2-simplex (a triangle in 3D space).
3. Individual histograms of each w component (w[0], w[1], w[2]) to show their marginal distributions.
4. Histogram of m showing that it follows a uniform distribution.

These visualizations help understand how AugMix linearly combines multiple augmentations
with random weights that respect convexity (sum to 1) and further blend with the original image.
"""

# 1. Check that w sums to 1
w_sums = w_samples.sum(axis=1)

plt.figure(figsize=(6, 4))
plt.hist(w_sums, bins=50, color='green', alpha=0.7)
plt.title('Sum of Dirichlet Samples (Should be 1)')
plt.xlabel('Sum')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 2. 3D Scatter plot of (w0, w1, w2)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w_samples[:, 0], w_samples[:, 1], w_samples[:, 2], alpha=0.3, s=5)
ax.set_title('Dirichlet(1,1,1) Samples in 3D (Simplex)')
ax.set_xlabel('w[0]')
ax.set_ylabel('w[1]')
ax.set_zlabel('w[2]')
plt.show()

# 3. Individual histograms of w[0], w[1], w[2]
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i in range(3):
    axs[i].hist(w_samples[:, i], bins=50, alpha=0.7)
    axs[i].set_title(f'Distribution of w[{i}]')
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Histogram of m (Beta(1,1))
plt.figure(figsize=(6, 4))
plt.hist(m_samples, bins=50, color='orange', alpha=0.7)
plt.title('Distribution of m (Beta(1,1))')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
