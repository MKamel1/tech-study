import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.suptitle('Central Limit Theorem: Sample Means Converge to Normal',
             fontsize=16, fontweight='bold')

# Three very non-Normal source distributions
sources = [
    ('Exponential(1)', lambda size: np.random.exponential(1, size)),
    ('Uniform(0,1)', lambda size: np.random.uniform(0, 1, size)),
    ('Bernoulli(0.3)', lambda size: np.random.binomial(1, 0.3, size))
]

sample_sizes = [1, 5, 30, 100]
n_simulations = 10000

for row, (name, sampler) in enumerate(sources):
    for col, n in enumerate(sample_sizes):
        # Simulate n_simulations sample means, each from n observations
        means = [sampler(n).mean() for _ in range(n_simulations)]
        
        ax = axes[row, col]
        ax.hist(means, bins=50, density=True, alpha=0.7, color='#4a90d9', edgecolor='black')
        
        # Overlay theoretical Normal
        mu = np.mean(means)
        sigma = np.std(means)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
        if col == 0:
            ax.set_ylabel(name, fontsize=12, fontweight='bold')
        if row == 0:
            ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        if row == 0 and col == 3:
            ax.legend()

plt.tight_layout()
plt.savefig('clt_convergence.png', dpi=150, bbox_inches='tight')
# plt.show()
