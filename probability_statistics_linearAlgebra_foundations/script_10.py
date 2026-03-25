import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Law of Large Numbers: Running Average Converges to True Mean',
             fontsize=16, fontweight='bold')

distributions = [
    ('Exponential(2)', np.random.exponential, {'scale': 2}, 2.0),
    ('Bernoulli(0.7)', np.random.binomial, {'n': 1, 'p': 0.7}, 0.7),
    ('Uniform(0, 10)', np.random.uniform, {'low': 0, 'high': 10}, 5.0)
]

N = 5000

for ax, (name, dist_func, params, true_mean) in zip(axes, distributions):
    # Multiple independent runs to show convergence
    for run in range(5):
        samples = dist_func(size=N, **params)
        running_avg = np.cumsum(samples) / np.arange(1, N + 1)
        ax.plot(running_avg, alpha=0.5, linewidth=0.8)
    
    ax.axhline(y=true_mean, color='red', linewidth=2, linestyle='--',
               label=f'True mean = {true_mean}')
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('Number of samples (n)')
    ax.set_ylabel('Running average')
    ax.legend()
    ax.set_xlim(0, N)

plt.tight_layout()
plt.savefig('lln_convergence.png', dpi=150, bbox_inches='tight')
# plt.show()
