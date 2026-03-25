import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Multivariate Normal: Effect of Covariance', fontsize=16, fontweight='bold')

# Create grid
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Three different covariance matrices
covs = [
    ([[1, 0], [0, 1]], 'Spherical\nSigma = I'),
    ([[2, 0], [0, 0.5]], 'Axis-aligned\nSigma = diag(2, 0.5)'),
    ([[2, 1.2], [1.2, 1]], 'Correlated\nSigma = [[2,1.2],[1.2,1]]')
]

for ax, (cov, title) in zip(axes, covs):
    rv = stats.multivariate_normal([0, 0], cov)
    ax.contour(X, Y, rv.pdf(pos), levels=8, cmap='Blues')
    ax.contourf(X, Y, rv.pdf(pos), levels=8, cmap='Blues', alpha=0.4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig('multivariate_normal_contours.png', dpi=150, bbox_inches='tight')
# plt.show()
