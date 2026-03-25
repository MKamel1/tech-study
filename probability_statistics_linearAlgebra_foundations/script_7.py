import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Continuous Distributions Gallery', fontsize=16, fontweight='bold')

# --- 1. Normal: 68-95-99.7 rule ---
x = np.linspace(-4, 4, 300)
pdf = stats.norm.pdf(x)
axes[0, 0].plot(x, pdf, 'k-', linewidth=2)
for sigma, color, alpha in [(1, '#4a90d9', 0.4), (2, '#50c878', 0.25), (3, '#ffa500', 0.15)]:
    mask = (x >= -sigma) & (x <= sigma)
    axes[0, 0].fill_between(x[mask], pdf[mask], color=color, alpha=alpha,
                            label=f'+/-{sigma}sigma: {stats.norm.cdf(sigma)-stats.norm.cdf(-sigma):.1%}')
axes[0, 0].set_title('Normal: 68-95-99.7 Rule', fontsize=12)
axes[0, 0].legend(fontsize=8)

# --- 2. t vs Normal ---
x = np.linspace(-5, 5, 300)
axes[0, 1].plot(x, stats.norm.pdf(x), 'k-', linewidth=2, label='Normal')
for df, color in [(1, '#ff6b6b'), (3, '#ffa500'), (10, '#4a90d9')]:
    axes[0, 1].plot(x, stats.t.pdf(x, df), '--', color=color, linewidth=1.5, label=f't(df={df})')
axes[0, 1].set_title('t-Distribution vs Normal', fontsize=12)
axes[0, 1].legend()

# --- 3. Chi-squared (varying df) ---
x = np.linspace(0.01, 25, 300)
for df, color in [(1, '#ff6b6b'), (3, '#ffa500'), (5, '#4a90d9'), (10, '#50c878')]:
    axes[0, 2].plot(x, stats.chi2.pdf(x, df), linewidth=2, color=color, label=f'df={df}')
axes[0, 2].set_title('Chi-squared(k)', fontsize=12)
axes[0, 2].legend()

# --- 4. Exponential (varying lambda) ---
x = np.linspace(0, 5, 300)
for lam, color in [(0.5, '#4a90d9'), (1, '#50c878'), (2, '#ffa500')]:
    axes[1, 0].plot(x, stats.expon.pdf(x, scale=1/lam), linewidth=2, color=color,
                    label=f'lambda={lam}')
axes[1, 0].set_title('Exponential(lambda)', fontsize=12)
axes[1, 0].legend()

# --- 5. Log-Normal ---
x = np.linspace(0.01, 10, 300)
for sigma, color in [(0.25, '#4a90d9'), (0.5, '#50c878'), (1.0, '#ffa500')]:
    axes[1, 1].plot(x, stats.lognorm.pdf(x, s=sigma), linewidth=2, color=color,
                    label=f'sigma={sigma}')
axes[1, 1].set_title('Log-Normal(0, sigma)', fontsize=12)
axes[1, 1].legend()

# --- 6. F-distribution ---
x = np.linspace(0.01, 6, 300)
for d1, d2, color in [(1, 1, '#ff6b6b'), (5, 5, '#4a90d9'), (10, 30, '#50c878')]:
    axes[1, 2].plot(x, stats.f.pdf(x, d1, d2), linewidth=2, color=color,
                    label=f'F({d1},{d2})')
axes[1, 2].set_title('F-Distribution', fontsize=12)
axes[1, 2].legend()

for ax in axes.flat:
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

plt.tight_layout()
plt.savefig('continuous_distributions_gallery.png', dpi=150, bbox_inches='tight')
# plt.show()
