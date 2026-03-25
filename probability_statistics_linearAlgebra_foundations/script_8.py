import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Modeling Real-World Continuous Phenomena', fontsize=16, fontweight='bold')

# --- 1. Human Heights / Measurement Errors (Normal) ---
# Sum of many small genetic/environmental factors -> Normal via CLT
true_mu, true_sigma = 170, 7.5
heights = np.random.normal(true_mu, true_sigma, size=2000)
x_norm = np.linspace(140, 200, 200)
pdf_norm = stats.norm.pdf(x_norm, true_mu, true_sigma)

axes[0, 0].hist(heights, bins=40, density=True, color='#4a90d9', alpha=0.6, 
                edgecolor='black', label='Observed Heights (cm)')
axes[0, 0].plot(x_norm, pdf_norm, 'r-', linewidth=3, label=f'Normal fit ($\mu$=170, $\sigma$=7.5)')
axes[0, 0].set_title('Adult Heights (Sum of Many Effects)', fontsize=12)
axes[0, 0].set_xlabel('Height (cm)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# --- 2. Server Time-to-Failure (Exponential) ---
# Constant hazard rate (no aging)
failure_rate = 1/50 # 1 failure per 50 days on avg
time_to_failure = np.random.exponential(scale=1/failure_rate, size=1000)
x_exp = np.linspace(0, 300, 200)
pdf_exp = stats.expon.pdf(x_exp, scale=1/failure_rate)

axes[0, 1].hist(time_to_failure, bins=40, density=True, color='#50c878', alpha=0.6,
                edgecolor='black', label='Observed Times until Failure')
axes[0, 1].plot(x_exp, pdf_exp, 'r-', linewidth=3, label=f'Exponential fit (mean=50)')
axes[0, 1].set_title('Server Cluster: Time Until Next Failure', fontsize=12)
axes[0, 1].set_xlabel('Days')
axes[0, 1].legend()

# --- 3. Income / House Prices (Log-Normal) ---
# Multiplicative compounding effects
# Median income ~50k, but severe right skew
log_mu, log_sigma = np.log(50), 0.6 
incomes = np.random.lognormal(mean=log_mu, sigma=log_sigma, size=2000)
x_logn = np.linspace(10, 200, 200)
pdf_logn = stats.lognorm.pdf(x_logn, s=log_sigma, scale=np.exp(log_mu))

axes[1, 0].hist(incomes, bins=50, range=(0, 200), density=True, color='#ffa500', alpha=0.6,
                edgecolor='black', label='Observed Income Data')
axes[1, 0].plot(x_logn, pdf_logn, 'r-', linewidth=3, label=f'Log-Normal fit')
axes[1, 0].set_title('Annual Incomes (Multiplicative Effects)', fontsize=12)
axes[1, 0].set_xlabel('Income ($1000s)')
axes[1, 0].legend()

# --- 4. A/B Test Conversion Rates (Beta) ---
# Probabilities bounded between [0, 1]
# E.g., prior belief about a true CTR after seeing 40 clicks and 960 no-clicks
alpha_prior, beta_prior = 40, 960
ctr_samples = np.random.beta(a=alpha_prior, b=beta_prior, size=2000)
x_beta = np.linspace(0.01, 0.08, 200)
pdf_beta = stats.beta.pdf(x_beta, a=alpha_prior, b=beta_prior)

axes[1, 1].hist(ctr_samples, bins=40, density=True, color='#9370db', alpha=0.6,
                edgecolor='black', label='Sampled CTR Probabilities')
axes[1, 1].plot(x_beta, pdf_beta, 'r-', linewidth=3, label=f'Beta fit ($\\alpha$=40, $\\beta$=960)')
axes[1, 1].set_title('Uncertainty over True CTR (Bounded)', fontsize=12)
axes[1, 1].set_xlabel('Click-Through Rate Probability')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('continuous_phenomena.png', dpi=150, bbox_inches='tight')
# plt.show()
