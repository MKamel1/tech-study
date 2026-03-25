import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Improved Dirac Delta visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

mu = 0
x = np.linspace(-3, 3, 1000)

# Panel 1: Limit of Normal distributions
sigmas = [1.0, 0.5, 0.2, 0.05]
colors = ['#4a90d9', '#50c878', '#ffa500', '#ff6b6b']

for sig, col in zip(sigmas, colors):
    y = stats.norm.pdf(x, mu, sig)
    ax1.plot(x, y, color=col, linewidth=2, label=f'$\\sigma = {sig}$')
    ax1.fill_between(x, y, alpha=0.1, color=col)

ax1.set_xlim(-3, 3)
ax1.set_ylim(0, 8)
ax1.set_title('Limit of Normal Distributions\n$\\lim_{\\sigma \\to 0} \\mathcal{N}(0, \\sigma^2)$', fontsize=12, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.legend()

# Panel 2: Standard Impulse Representation
ax2.axhline(0, color='gray', linewidth=1)
# Draw the impulse arrow
ax2.annotate('', xy=(mu, 1), xytext=(mu, 0),
            arrowprops=dict(facecolor='#ff6b6b', shrink=0, width=3, headwidth=10))
ax2.plot(mu, 0, 'ko', markersize=6)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-0.1, 1.2)
ax2.set_title('Standard Impulse Representation\n$p(x) = \\delta(x)$', fontsize=12, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('Probability Mass')
ax2.text(mu + 0.2, 0.5, 'Area = 1', fontsize=11, color='#ff6b6b', fontweight='bold')

plt.tight_layout()
plt.savefig('dirac_delta.png', dpi=150, bbox_inches='tight')
plt.close()

print("Improved Dirac Delta graph generated!")
