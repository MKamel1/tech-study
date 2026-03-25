import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Discrete Distributions Gallery', fontsize=16, fontweight='bold')

# --- 1. Bernoulli ---
p = 0.3
x = [0, 1]
axes[0, 0].bar(x, [1-p, p], color=['#ff6b6b', '#4a90d9'], width=0.4, edgecolor='black')
axes[0, 0].set_title(f'Bernoulli(p={p})', fontsize=12)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Failure (0)', 'Success (1)'])
axes[0, 0].set_ylabel('P(X = x)')
axes[0, 0].set_ylim(0, 1)

# --- 2. Binomial (varying n) ---
for n, color in [(10, '#4a90d9'), (20, '#50c878'), (50, '#ffa500')]:
    x_binom = np.arange(0, n + 1)
    axes[0, 1].bar(x_binom, stats.binom.pmf(x_binom, n, 0.3),
                   alpha=0.5, label=f'n={n}, p=0.3', color=color)
axes[0, 1].set_title('Binomial(n, p=0.3)', fontsize=12)
axes[0, 1].set_xlabel('k (successes)')
axes[0, 1].set_ylabel('P(X = k)')
axes[0, 1].legend()

# --- 3. Poisson (varying lambda) ---
for lam, color in [(1, '#4a90d9'), (4, '#50c878'), (10, '#ffa500')]:
    x_pois = np.arange(0, 25)
    axes[0, 2].bar(x_pois, stats.poisson.pmf(x_pois, lam),
                   alpha=0.5, label=f'lambda={lam}', color=color)
axes[0, 2].set_title('Poisson(lambda)', fontsize=12)
axes[0, 2].set_xlabel('k')
axes[0, 2].set_ylabel('P(X = k)')
axes[0, 2].legend()

# --- 4. Geometric ---
p_geo = 0.3
x_geo = np.arange(1, 16)
axes[1, 0].bar(x_geo, stats.geom.pmf(x_geo, p_geo), color='#9370db',
               alpha=0.8, edgecolor='black')
axes[1, 0].set_title(f'Geometric(p={p_geo})', fontsize=12)
axes[1, 0].set_xlabel('k (trials until success)')
axes[1, 0].set_ylabel('P(X = k)')
axes[1, 0].axvline(x=1/p_geo, color='red', linestyle='--', label=f'E[X]={1/p_geo:.1f}')
axes[1, 0].legend()

# --- 5. Negative Binomial (varying r) ---
for r, color in [(1, '#4a90d9'), (3, '#50c878'), (5, '#ffa500')]:
    x_nb = np.arange(r, r + 20)
    axes[1, 1].bar(x_nb, stats.nbinom.pmf(x_nb - r, r, 0.4),
                   alpha=0.5, label=f'r={r}, p=0.4', color=color)
axes[1, 1].set_title('Negative Binomial(r, p=0.4)', fontsize=12)
axes[1, 1].set_xlabel('k (trials until r-th success)')
axes[1, 1].set_ylabel('P(X = k)')
axes[1, 1].legend()

# --- 6. Poisson vs Binomial approximation ---
n_approx, p_approx = 100, 0.03
lam_approx = n_approx * p_approx
x_approx = np.arange(0, 15)
axes[1, 2].bar(x_approx - 0.15, stats.binom.pmf(x_approx, n_approx, p_approx),
               width=0.3, label=f'Binomial({n_approx}, {p_approx})', color='#4a90d9', alpha=0.8)
axes[1, 2].bar(x_approx + 0.15, stats.poisson.pmf(x_approx, lam_approx),
               width=0.3, label=f'Poisson({lam_approx})', color='#ff6b6b', alpha=0.8)
axes[1, 2].set_title('Poisson Approximation to Binomial', fontsize=12)
axes[1, 2].set_xlabel('k')
axes[1, 2].set_ylabel('P(X = k)')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('discrete_distributions_gallery.png', dpi=150, bbox_inches='tight')
# plt.show()
