import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Modeling Real-World Discrete Phenomena', fontsize=16, fontweight='bold')

# --- 1. Website Conversions (Binomial) ---
n_visitors = 100
true_cvr = 0.05
# Simulate 1000 days of website traffic (each day has 100 visitors)
daily_conversions = np.random.binomial(n=n_visitors, p=true_cvr, size=1000)
x_bin = np.arange(0, 15)
pmf_bin = stats.binom.pmf(x_bin, n=n_visitors, p=true_cvr)

axes[0, 0].hist(daily_conversions, bins=np.arange(-0.5, 15.5, 1), density=True, 
                color='#4a90d9', alpha=0.6, edgecolor='black', label='Observed Data (1000 days)')
axes[0, 0].plot(x_bin, pmf_bin, 'ro-', linewidth=2, label=f'Binomial fit (n=100, p=0.05)')
axes[0, 0].set_title('Daily Ad Conversions', fontsize=12)
axes[0, 0].set_xlabel('Number of Conversions per Day')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].legend()

# --- 2. Customer Queue Arrivals (Poisson) ---
true_rate = 12 # 12 customers per hour
# Simulate arrivals per hour for 1000 hours
hourly_arrivals = np.random.poisson(lam=true_rate, size=1000)
x_pois = np.arange(0, 30)
pmf_pois = stats.poisson.pmf(x_pois, mu=true_rate)

axes[0, 1].hist(hourly_arrivals, bins=np.arange(-0.5, 30.5, 1), density=True,
                color='#50c878', alpha=0.6, edgecolor='black', label='Observed Arrivals (1000 hrs)')
axes[0, 1].plot(x_pois, pmf_pois, 'ro-', linewidth=2, label=f'Poisson fit ($\lambda$=12)')
axes[0, 1].set_title('Customer Arrivals at a Store', fontsize=12)
axes[0, 1].set_xlabel('Arrivals per Hour')
axes[0, 1].legend()

# --- 3. Impressions until Click (Geometric) ---
ctr = 0.1 # 10% Click-Through Rate
# Simulate how many impressions 1000 different users need before they click
impressions = np.random.geometric(p=ctr, size=1000)
x_geom = np.arange(1, 40)
pmf_geom = stats.geom.pmf(x_geom, p=ctr)

axes[1, 0].hist(impressions, bins=np.arange(0.5, 40.5, 1), density=True,
                color='#ffa500', alpha=0.6, edgecolor='black', label='Observed Impressions to Click')
axes[1, 0].plot(x_geom, pmf_geom, 'ro-', linewidth=2, label=f'Geometric fit (p=0.1)')
axes[1, 0].set_title('Ad Impressions Until First Click', fontsize=12)
axes[1, 0].set_xlabel('Number of Impressions')
axes[1, 0].legend()

# --- 4. Quality Control Failures (Negative Binomial) ---
# Inspect items until finding 3 defective ones (defect rate = 5%)
r_defects = 3
p_defect = 0.05
# Note: scipy's nbinom expects number of *successes* before r failures, or vice versa depending on definition.
# Here we model extra non-defective items inspected before finding 3 defects.
extra_items = np.random.negative_binomial(n=r_defects, p=p_defect, size=1000)
total_items = extra_items + r_defects
x_nb = np.arange(r_defects, 150)
pmf_nb = stats.nbinom.pmf(x_nb - r_defects, r_defects, p_defect)

axes[1, 1].hist(total_items, bins=np.arange(r_defects-0.5, 150.5, 5), density=True,
                color='#9370db', alpha=0.6, edgecolor='black', label='Observed Items Inspected')
axes[1, 1].plot(x_nb, pmf_nb, 'r-', linewidth=2, label=f'Neg. Binom fit (r=3, p=0.05)')
axes[1, 1].set_title('Items Inspected to Find 3 Defects', fontsize=12)
axes[1, 1].set_xlabel('Total Items Inspected')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('discrete_phenomena.png', dpi=150, bbox_inches='tight')
# plt.show()
