import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 1. Multinomial
fig, ax = plt.subplots(figsize=(6, 4))
n_trials = 100
p_vals = [0.2, 0.5, 0.3]
sample = np.random.multinomial(n_trials, p_vals)
categories = ['Category A\n(p=0.2)', 'Category B\n(p=0.5)', 'Category C\n(p=0.3)']
ax.bar(categories, sample, color=['#4a90d9', '#50c878', '#ffa500'], edgecolor='black', alpha=0.8)
ax.set_title(f'Multinomial Sample (n={n_trials} trials)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(sample):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('multinomial_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Uniform
fig, ax = plt.subplots(figsize=(6, 4))
a, b = -2, 4
x = np.linspace(-4, 6, 400)
y = stats.uniform.pdf(x, loc=a, scale=b-a)
ax.plot(x, y, 'k-', linewidth=2)
ax.fill_between(x, y, where=((x >= a) & (x <= b)), color='#9370db', alpha=0.5)
ax.set_title(f'Uniform Distribution (a={a}, b={b})', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_ylim(0, 0.25)
plt.tight_layout()
plt.savefig('uniform_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Gamma
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 20, 400)
# Gamma PDF: scipy uses shape (a) and scale (1/beta)
for a, scale, color in [(1, 2.0, '#ff6b6b'), (2, 2.0, '#4a90d9'), (3, 2.0, '#50c878'), (5, 1.0, '#ffa500')]:
    y = stats.gamma.pdf(x, a=a, scale=scale)
    ax.plot(x, y, linewidth=2, color=color, label=f'$\\alpha={a}, \\beta={1/scale:.1f}$')
ax.set_title('Gamma Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig('gamma_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Dirac Delta
fig, ax = plt.subplots(figsize=(6, 4))
mu = 2.5
ax.axhline(0, color='gray', linewidth=1)
ax.annotate('', xy=(mu, 1), xytext=(mu, 0),
            arrowprops=dict(facecolor='#ff6b6b', shrink=0, width=3, headwidth=10))
ax.plot(mu, 0, 'ko', markersize=6)
ax.set_xlim(0, 5)
ax.set_ylim(-0.1, 1.2)
ax.set_yticks([])
ax.set_xticks([0, 1, 2, mu, 3, 4, 5])
ax.set_xticklabels(['0', '1', '2', r'$\mu=2.5$', '3', '4', '5'])
ax.set_title('Dirac Delta Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.text(mu + 0.1, 0.5, 'Area = 1', fontsize=10, color='#ff6b6b', fontweight='bold')
plt.tight_layout()
plt.savefig('dirac_delta.png', dpi=150, bbox_inches='tight')
plt.close()

print("Graphs generated safely!")
