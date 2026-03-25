import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Beta Distribution Shape Gallery', fontsize=16, fontweight='bold')

params = [
    (1, 1, 'Uniform: Beta(1,1)'),
    (0.5, 0.5, 'U-shaped: Beta(0.5,0.5)'),
    (2, 5, 'Left-skewed: Beta(2,5)'),
    (5, 2, 'Right-skewed: Beta(5,2)'),
    (5, 5, 'Symmetric: Beta(5,5)'),
    (50, 50, 'Concentrated: Beta(50,50)')
]

x = np.linspace(0.001, 0.999, 300)

for ax, (a, b, title) in zip(axes.flat, params):
    pdf = stats.beta.pdf(x, a, b)
    ax.plot(x, pdf, color='#4a90d9', linewidth=2.5)
    ax.fill_between(x, pdf, alpha=0.3, color='#4a90d9')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    mean = a / (a + b)
    ax.axvline(x=mean, color='red', linestyle='--', alpha=0.7, label=f'Mean={mean:.2f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('beta_distribution_gallery.png', dpi=150, bbox_inches='tight')
# plt.show()
