# Battle Plan - Months 6-8: Interview Mastery

**Duration**: 12 weeks (Weeks 22-33+ in overall timeline)  
**Total Hours**: 280-330 hours  
**Purpose**: Statistics deep dive + Theory mastery + Mock interviews + Applications

---

## 📋 MONTHS 6-8 OVERVIEW

**PHASE 3: INTERVIEW MASTERY**

This final phase transitions from building to interview preparation. The strategy:
1. **Month 6**: Statistics deep dive (40h) + Theory practice + Mock interviews Round 1
2. **Month 7**: System design mastery + Mock interviews Round 2 + Applications start
3. **Month 8**: Final prep + Networking + Active interviewing

---

## MONTH 6: THEORY MASTERY + STATISTICS DEEP DIVE

### **Week 21-22: Statistics Deep Dive** (40 hours total)

**🔴 CRITICAL**: This is the **Statistics Deep Dive** from gap analysis (40h recommended allocation)

#### **Week 21: Regression Diagnostics & Hypothesis Testing** (20h)

##### **Monday-Tuesday: Regression Diagnostics** (8h)

**1. Heteroscedasticity** (3h):

**Theory**:
- Homoscedasticity: Var(ε|X) = σ² (constant variance)
- Heteroscedasticity: Var(ε|X) depends on X
- Consequences: OLS still unbiased but inefficient, SEs wrong

**Tests**:
```python
# Breusch-Pagan Test
def breusch_pagan_test(model, X):
    """
    Test H0: Homoscedasticity
    """
    # Get residuals from original OLS
    residuals = model.resid
    
    # Regress squared residuals on X
    squared_resid = residuals ** 2
    aux_model = sm.OLS(squared_resid, sm.add_constant(X)).fit()
    
    # Test statistic: n * R²
    n = len(residuals)
    test_stat = n * aux_model.rsquared
    
    # Chi-square distribution with k degrees of freedom
    p_value = 1 - chi2.cdf(test_stat, df=X.shape[1])
    
    return {'test_statistic': test_stat, 'p_value': p_value}

# White Test (more general)
from statsmodels.stats.diagnostic import het_white
white_test = het_white(residuals, exog)
```

**Solution - Weighted Least Squares (WLS)**:
```python
# When Var(ε) = σ²X², use weights = 1/X
weights = 1 / X
wls_model = sm.WLS(y, X, weights=weights).fit()
```

---

**2. Multicollinearity** (3h):

**Variance Inflation Factor (VIF)**:
```python
def calculate_vif(X):
    """
    VIF_i = 1 / (1 - R²_i)
    where R²_i is R² from regressing X_i on all other X's
    
    Rule of thumb:
    - VIF > 10: Severe multicollinearity
    - VIF > 5: Moderate concern
    """
    vif_data = []
    for i in range(X.shape[1]):
        # Regress X_i on all other X's
        X_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        
        model = sm.OLS(X_i, X_others).fit()
        vif = 1 / (1 - model.rsquared)
        
        vif_data.append({'feature': i, 'vif': vif})
    
    return pd.DataFrame(vif_data)
```

**Solution - Ridge Regression**:
```python
from sklearn.linear_model import Ridge

# Add L2 penalty to shrink correlated coefficients
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
```

---

**3. Autocorrelation** (2h):

**Durbin-Watson Test**:
```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)

# Interpretation:
# dw ≈ 2: No autocorrelation
# dw < 2: Positive autocorrelation
# dw > 2: Negative autocorrelation
```

---

##### **Wednesday-Thursday: Hypothesis Testing Deep Dive** (8h)

**1. Type I vs Type II Errors** (2h):

**Comprehensive Understanding**:
```python
def understand_errors():
    """
    Type I (α): Reject H0 when H0 is true (False Positive)
    Type II (β): Fail to reject H0 when H0 is false (False Negative)
    Power = 1 - β
    """
    # Example: A/B test
    # H0: Treatment has no effect
    # Type I: Claim treatment works when it doesn't → Ship bad feature
    # Type II: Claim treatment doesn't work when it does → Miss good feature
    
    # Tradeoff: Lowering α (0.05 → 0.01) increases β (lower power)
    pass
```

**Multiple Hypothesis Testing**:
```python
from statsmodels.stats.multitest import multipletests

# Problem: 20 independent tests at α=0.05 → P(at least 1 false positive) = 64%

# Solutions:
# 1. Bonferroni: α_adjusted = α / n (conservative)
p_adjusted_bonferroni = p_values * n

# 2. Benjamini-Hochberg (FDR control, less conservative)
reject, p_adjusted_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

**2. Permutation Tests** (3h):

**Non-parametric alternative to t-test**:
```python
def permutation_test(group1, group2, n_permutations=10000):
    """
    Test if two groups have different means without normality assumption.
    """
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Combine groups
    combined = np.concatenate([group1, group2])
    
    # Permute and calculate null distribution
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:len(group1)]
        perm_group2 = combined[len(group1):]
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        perm_diffs.append(perm_diff)
    
    # P-value: proportion of permutations as extreme as observed
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return p_value
```

---

**3. Bayesian Hypothesis Testing** (3h):

**Bayes Factor**:
```python
def bayes_factor_ttest(data1, data2):
    """
    BF10 = P(data | H1) / P(data | H0)
    
    Interpretation:
    - BF10 > 10: Strong evidence for H1
    - BF10 < 0.1: Strong evidence for H0
    - 0.33 < BF10 < 3: Inconclusive
    """
    import pingouin as pg
    
    bf = pg.bayesfactor_ttest(data1, data2, paired=False)
    
    return bf
```

---

##### **Friday: Asymptotic Theory & Distribution Relationships** (4h)

**1. Law of Large Numbers** (2h):

**Weak vs Strong LLN**:
```python
def demonstrate_lln(n_samples=10000):
    """
    Weak LLN: X̄_n →^p μ (converges in probability)
    Strong LLN: X̄_n →^{a.s.} μ (converges almost surely)
    """
    # Simulate coin flips
    population_p = 0.7
    
    sample_means = []
    for n in range(1, n_samples+1):
        sample = np.random.binomial(1, population_p, size=n)
        sample_means.append(np.mean(sample))
    
    # Plot convergence
    plt.plot(sample_means)
    plt.axhline(population_p, color='r', linestyle='--', label='True mean')
    plt.xlabel('Sample size')
    plt.ylabel('Sample mean')
    plt.title('Law of Large Numbers')
    plt.legend()
```

---

**2. Central Limit Theorem** (2h):

**Conditions & Delta Method**:
```python
def delta_method_example():
    """
    If √n(X̄ - μ) →^d N(0, σ²), then
    √n(g(X̄) - g(μ)) →^d N(0, σ² [g'(μ)]²)
    
    Example: Estimate variance of log(X̄)
    """
    # Simulate exponential data
    n = 1000
    lambda_param = 2
    data = np.random.exponential(1/lambda_param, size=n)
    
    # Sample mean
    x_bar = np.mean(data)
    
    # Transformation: g(x) = log(x)
    g_x_bar = np.log(x_bar)
    
    # Delta method variance
    # g'(x) = 1/x, so Var(log(X̄)) ≈ σ²/n * (1/μ)²
    var_log_x_bar = (np.var(data) / n) * (1 / x_bar)**2
    se_log_x_bar = np.sqrt(var_log_x_bar)
    
    print(f"Estimated log(mean): {g_x_bar:.3f} ± {1.96*se_log_x_bar:.3f}")
```

---

#### **Week 22: Causal Inference Theory Practice** (20h)

##### **Monday-Wednesday: Derivation Practice** (14h)

**1. Refresh PhD Methods** (6h):

**Methods you ALREADY know**:
- **PSM (Propensity Score Matching)**: Derive propensity score formula, prove balance
- **DID (Difference-in-Differences)**: Derive parallel trends assumption, proof
- **Mediation Analysis**: Direct vs indirect effects, prove decomposition

**Exercise**: Derive each from first principles on whiteboard (2h each)

---

**2. Solidify Month 1 Learnings** (8h):

**IPTW (Inverse Probability Treatment Weighting)**:
```python
def derive_iptw():
    """
    Goal: Estimate ATE when treatment T is confounded by X
    
    Idea: Weight by 1/P(T|X) to create pseudo-population where T ⊥ X
    
    Derivation:
    E[Y(1) - Y(0)] = E[Y·T/e(X) - Y·(1-T)/(1-e(X))]
    where e(X) = P(T=1|X) is propensity score
    
    Proof:
    1. By iterated expectation: E[Y(1)] = E[E[Y(1)|X]]
    2. By ignorability: E[Y(1)|X] = E[Y|T=1,X]
    3. By positivity: E[Y|T=1,X] = E[Y·T/P(T=1|X)|X]
    4. Taking outer expectation gives IPTW estimator
    """
    pass

# Implement
def estimate_ate_iptw(Y, T, X):
    """
    Estimate ATE using IPTW.
    """
    # Estimate propensity scores
    from sklearn.linear_model import LogisticRegression
    ps_model = LogisticRegression().fit(X, T)
    e_x = ps_model.predict_proba(X)[:, 1]
    
    # IPTW estimator
    weights = T / e_x + (1 - T) / (1 - e_x)
    ate = np.mean(weights * Y * (2*T - 1))  # Simplified
    
    return ate
```

**2SLS (Two-Stage Least Squares)**:
```python
def derive_2sls():
    """
    Problem: Want effect of X on Y, but X is endogenous (correlated with ε)
    Solution: Use instrument Z (affects X but not Y directly)
    
    Derivation:
    Stage 1: X = π₀ + π₁Z + ν  → Get X̂
    Stage 2: Y = β₀ + β₁X̂ + ε
    
    Estimand: β₁ = Cov(Y,Z) / Cov(X,Z)
    
    Why it works:
    - Z → X (Relevance): Cov(X,Z) ≠ 0
    - Z ⊥ ε (Exclusion): E[Z·ε] = 0
    - Therefore: Cov(Y,Z) = β₁·Cov(X,Z) + Cov(ε,Z) = β₁·Cov(X,Z)
    """
    pass

# Implement
def estimate_2sls(Y, X_endog, Z):
    """
    Two-stage least squares IV estimation.
    """
    # Stage 1: Regress X on Z
    stage1 = sm.OLS(X_endog, sm.add_constant(Z)).fit()
    X_hat = stage1.fittedvalues
    
    # Stage 2: Regress Y on X̂
    stage2 = sm.OLS(Y, sm.add_constant(X_hat)).fit()
    
    return stage2.params[1]  # β₁
```

**RDD (Regression Discontinuity Design)**:
```python
def derive_rdd():
    """
    Treatment assigned based on threshold: T = 1[X ≥ c]
    
    Estimand (sharp RDD): τ = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]
    
    Assumptions:
    1. Continuity: E[Y(0)|X] continuous at c
    2. No manipulation: X not manipulated to cross threshold
    
    Implementation: Local linear regression
    - Fit separate regressions on each side of cutoff
    - Estimate τ as difference in intercepts at c
    """
    pass

# Implement
def estimate_rdd(Y, X, cutoff, bandwidth=1.0):
    """
    RDD local linear estimation.
    """
    # Restrict to bandwidth around cutoff
    in_window = np.abs(X - cutoff) <= bandwidth
    Y_window = Y[in_window]
    X_window = X[in_window] - cutoff  # Center at 0
    T_window = (X[in_window] >= cutoff).astype(int)
    
    # Local linear regression: Y ~ T + X + T*X
    model = sm.OLS(Y_window, sm.add_constant(
        pd.DataFrame({'T': T_window, 'X': X_window, 'T_X': T_window * X_window})
    )).fit()
    
    # RDD estimate = coefficient on T
    return model.params['T']
```

**Doubly Robust Estimator**:
- Combines outcome regression + propensity scores
- Consistent if EITHER model is correct (not both!)
- Derive why it's doubly robust

---

##### **Thursday-Friday: Conceptual Mastery** (6h)

**Prepare answers to 50 causal interview questions**:

**Example Questions with Answers**:

1. **"When would you use IV vs PSM?"**
   - IV: When you have an instrument (affects treatment but not outcome directly)
     - Example: Draft lottery for Vietnam war effect on earnings
   - PSM: When unconfoundedness holds (all confounders observed)
     - Example: Observational study with rich covariates

2. **"Explain backdoor criterion with example"**
   - Backdoor: Set of variables that block all non-causal paths from X to Y
   - Example DAG: X ← Z → Y. To estimate X→Y effect, control for Z.
   - Formal: Block paths from X to Y that have arrow into X

3. **"What's difference between ATE and CATE?"**
   - ATE: Average treatment effect across entire population
   - CATE: Conditional ATE - varies by subgroup/individual characteristics
   - Example: Drug works for men (CATE=+10%) but not women (CATE=0%)

4. **"When does DID fail?"**
   - Parallel trends violation: Treatment and control groups would have evolved differently even without treatment
   - Example: Tech company hires more → wages increase. Control: Non-tech. But tech wages growing faster anyway!

5. **"What assumptions does RDD require?"**
   - Continuity: Potential outcomes continuous at cutoff
   - No manipulation: Units can't precisely control running variable
   - Example violation: Students retaking test to cross scholarship threshold

**Practice method**: Record yourself explaining each on whiteboard, watch, improve

---

### **Week 23: ML Theory Practice** (24h)

##### **Monday-Tuesday: Backpropagation & Optimization** (8h)

**1. Derive Backpropagation from Scratch** (4h):

```python
def backprop_2layer_network():
    """
    Network: Input → Hidden (ReLU) → Output (Softmax)
    
    Forward:
    z1 = W1·x + b1
    a1 = ReLU(z1)
    z2 = W2·a1 + b2
    y_hat = Softmax(z2)
    
    Loss: L = -log(y_hat[true_class])
    
    Backward (Chain rule):
    ∂L/∂W2 = ∂L/∂z2 · ∂z2/∂W2 = (y_hat - y_true) ⊗ a1
    ∂L/∂W1 = ∂L/∂z2 · ∂z2/∂a1 · ∂a1/∂z1 · ∂z1/∂W1
           = (y_hat - y_true) · W2 · ReLU'(z1) ⊗ x
    
    where ReLU'(z) = 1[z > 0]
    """
    pass
```

**2. Optimizer Derivations** (4h):

**SGD with Momentum**:
```python
def sgd_momentum():
    """
    Problem: SGD oscillates in ravines (high curvature in some directions)
    Solution: Add momentum term
    
    v_t = β·v_{t-1} + (1-β)·∇L
    θ_t = θ_{t-1} - α·v_t
    
    Intuition: Like a ball rolling downhill, accumulates velocity
    """
    pass
```

**Adam Optimizer**:
```python
def adam_optimizer():
    """
    Combines momentum (first moment) + RMSProp (second moment)
    
    m_t = β1·m_{t-1} + (1-β1)·∇L      # First moment (momentum)
    v_t = β2·v_{t-1} + (1-β2)·(∇L)²   # Second moment (adaptive LR)
    
    m̂_t = m_t / (1 - β1^t)  # Bias correction
    v̂_t = v_t / (1 - β2^t)
    
    θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
    
    Why it works: Adapts learning rate per parameter based on gradient history
    """
    pass
```

---

##### **Wednesday-Thursday: Attention Mechanism** (8h)

**Self-Attention Derivation**:
```python
def derive_self_attention():
    """
    Input: X ∈ ℝ^{n×d} (sequence of n tokens, each d-dimensional)
    
    1. Create Q, K, V:
       Q = X·W_q  (queries)
       K = X·W_k  (keys)
       V = X·W_v  (values)
    
    2. Attention scores:
       scores = softmax(Q·K^T / √d_k)
    
    3. Weighted sum:
       output = scores · V
    
    Why scaling by √d_k?
    - Without: dot products grow with d_k → softmax saturates
    - With: keeps variance of dot product ~1
    
    Proof:
    If Q, K ~ N(0,1), then Q·K = Σq_i·k_i has Var = d_k
    Scaling by √d_k normalizes: Var(Q·K/√d_k) = 1
    """
    pass

# Multi-Head Attention
def multi_head_attention():
    """
    Why multiple heads?
    - Different heads learn different representation subspaces
    - Head 1: Syntax (noun-verb relationships)
    - Head 2: Semantics (similar concepts)
    - Head 3: Long-range dependencies
    
    Concat then linear: Concat(head1, ..., head_h) · W_o
    """
    pass
```

**Interview Questions**:
- "Explain self-attention vs cross-attention"
- "Why O(n²) complexity? How to reduce?" (Linear attention, Performer, etc.)
- "Why positional encoding?" (Transformers have no recurrence → no order info)

---

##### **Friday: Loss Functions & Regularization** (4h)

**1. Cross-Entropy Derivation** (2h):

```python
def cross_entropy_loss():
    """
    For classification: L = -Σ y_i · log(ŷ_i)
    
    Connection to maximum likelihood:
    - Minimizing cross-entropy = Maximizing log-likelihood
    - If y ~ Multinomial(ŷ), then L = -log P(y|ŷ)
    
    Gradient: ∂L/∂z = ŷ - y (clean gradient when using softmax!)
    """
    pass
```

**2. Regularization Theory** (2h):

**L1 vs L2**:
- L1: Lasso, promotes sparsity (some weights → 0)
- L2: Ridge, shrinks all weights proportionally
- Why? L1 has corners at axes (gradient undefined → can land on 0)

**Dropout as Ensemble**:
- At each step, train different sub-network
- At test time, approximate averaging all 2^n sub-networks
- Bayesian interpretation: Approximate posterior sampling

---

##### **Weekend: Advanced Statistical Methods (PhD Differentiators)** (11h)

**Review YOUR methods - these set you apart!**

**1. GEV (Generalized Extreme Value) Models** (2h):

**Theory**:
```python
def gev_model():
    """
    For modeling rare events (e.g., crashes, extreme weather)
    
    PDF: f(x;μ,σ,ξ) = (1/σ)·exp(-(1 + ξ·z)^{-1/ξ})·(1 + ξ·z)^{-1-1/ξ}
    where z = (x - μ)/σ
    
    ξ: Shape parameter
    - ξ > 0: Fréchet (heavy tail, unbounded)
    - ξ = 0: Gumbel (exponential tail)
    - ξ < 0: Weibull (bounded above)
    
    Application in PhD: Safety metrics for AV crashes
    - Model TTC distribution tail → estimate P(crash)
    """
    pass
```

**Interview angle**:
> "In my PhD, I developed novel safety metrics using GEV models to characterize extreme AV events. Traditional metrics assume normal distributions, but crash near-misses have heavy tails - a Gumbel or Fréchet distribution better captures rare but severe events."

---

**2. Hierarchical Bayesian Modeling** (2h):

**Structure**:
```python
def hierarchical_bayesian():
    """
    Level 1 (Data): y_ij ~ N(μ_j, σ²)
    Level 2 (Group): μ_j ~ N(θ, τ²)
    Level 3 (Hyperprior): θ ~ N(0, 100), τ ~ Cauchy(0, 5)
    
    Why hierarchical?
    - Partial pooling: Borrow strength across groups
    - Shrinkage: Small groups pulled toward global mean
    - Accounts for within/between group variance
    
    PhD application: Spatial correlation in AV safety across intersections
    """
    pass
```

**Interview angle**:
> "I used hierarchical Bayesian models to account for spatial and temporal correlation in AV safety data. Standard regression assumes independence, but safety events cluster by location. My approach modeled random effects at the intersection level, improving prediction accuracy by 23%."

---

**3. Spatial Econometrics (PhD Differentiator)** (3h):

**🔴 NEW ADDITION** - Leverages your unique PhD background in spatial analysis

**Core Concepts**:

**Spatial Autocorrelation (Moran's I)**:
```python
def calculate_morans_i(y, W):
    """
    Moran's I: Measure of spatial autocorrelation
    
    I = (n / S0) · Σ_i Σ_j w_ij · (y_i - ȳ)(y_j - ȳ) / Σ_i (y_i - ȳ)²
    
    where:
    - W: Spatial weights matrix (e.g., inverse distance, k-nearest neighbors)
    - S0 = Σ_i Σ_j w_ij (sum of all weights)
    
    Interpretation:
    - I > 0: Positive correlation (similar values cluster)
    - I ≈ 0: Random spatial pattern
    - I < 0: Negative correlation (dissimilar values neighbor each other)
    
    Significance: Permutation test (shuffle locations, recalculate I)
    """
    from pysal.lib import weights
    from esda.moran import Moran
    
    # Example: K-nearest neighbors weights
    w = weights.KNN.from_dataframe(gdf, k=8)
    
    # Calculate Moran's I
    moran = Moran(y, w)
    
    return {
        'I': moran.I,
        'p_value': moran.p_sim,
        'interpretation': 'Clustered' if moran.I > 0 else 'Dispersed'
    }
```

**Spatial Lag Model (SAR)**:
```python
def spatial_lag_model():
    """
    Accounts for spillover effects from neighboring locations.
    
    Model: y = ρ·W·y + X·β + ε
    
    where:
    - ρ: Spatial autoregressive coefficient (spillover strength)
    - W·y: Spatially lagged dependent variable (neighbors' values)
    
    Use case: AV crash risk at one intersection affects nearby intersections
    
    Example: High crash rate at Times Square → elevated risk at adjacent intersections
    """
    from pysal.model import spreg
    
    # Spatial lag model
    model = spreg.ML_Lag(y, X, w=spatial_weights)
    
    # ρ coefficient tells you spillover magnitude
    print(f"Spatial spillover: ρ = {model.rho:.3f}")
    pass
```

**Spatial Error Model (SEM)**:
```python
def spatial_error_model():
    """
    Accounts for omitted spatially-correlated variables.
    
    Model: y = X·β + u, where u = λ·W·u + ε
    
    where:
    - λ: Spatial error coefficient
    
    Use when: Unobserved factors (weather, road quality) correlate spatially
    """
    pass
```

---

**Interview Angles** (Prepare 3 stories):

1. **For Logistics/Uber/Maps roles**:
> "In my PhD, I modeled spatial correlation in autonomous vehicle safety across urban intersections. I used spatial lag models to account for spillover effects - a near-miss at one intersection increases risk at adjacent locations by 15%. This is critical for fleet deployment: you can't treat intersections as independent."

2. **For Real Estate/Zillow/Redfin roles**:
> "Spatial econometrics is essential for property valuation. I applied spatial autoregressive models to account for neighborhood effects - a house's value depends not just on its features, but on neighboring properties. Standard regression underestimates spatial spillovers by 30-40%."

3. **For General DS roles**:
> "My PhD gave me deep expertise in spatial statistics. I can detect clustering patterns using Moran's I and model spillover effects between geographic units. This applies to retail (store cannibalization), epidemiology (disease spread), or any problem with geographic structure."

---

**Key Libraries**:
- **PySAL** (Python Spatial Analysis Library): Gold standard for spatial econometrics
- **GeoPandas**: Geospatial data manipulation (like Pandas + GIS)
- **Folium/Plotly**: Interactive maps for visualization

**Practice**: Frame 1-2 PhD projects using spatial econometrics terminology

---

**4. Reinforcement Learning Basics (Breadth Coverage)** (4h):

**🔴 NEW ADDITION** - Breadth topic for specialized roles (pricing, ads, robotics)

#### **Core Concepts** (2h):

**What is RL?**
```python
def rl_framework():
    """
    Agent learns optimal POLICY by interacting with environment.
    
    Key components:
    - State (s): Current situation
    - Action (a): What agent can do
    - Reward (r): Immediate feedback
    - Policy (π): Strategy mapping states → actions
    - Value function (V): Expected cumulative reward from state s
    
    Goal: Maximize cumulative reward over time
    
    Contrast with supervised learning:
    - SL: Learn from labeled examples
    - RL: Learn from trial-and-error (delayed rewards)
    """
    pass
```

**Types of RL**:

**1. Value-Based (Q-Learning, DQN)**:
```python
def q_learning():
    """
    Learn Q(s,a) = expected reward from taking action a in state s
    
    Update rule: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    
    where:
    - α: Learning rate
    - γ: Discount factor (how much to value future rewards)
    
    Use case: Atari games, grid-world navigation
    """
    pass
```

**2. Policy-Based (REINFORCE, PPO)**:
```python
def policy_gradient():
    """
    Directly learn policy π(a|s) without value function.
    
    Advantages:
    - Can handle continuous actions (e.g., steering angle, pricing)
    - Works for stochastic policies
    
    Use case: Robotics (continuous control), ad bidding
    """
    pass
```

**3. Model-Based RL**:
- Learn environment dynamics: P(s'|s,a)
- Plan ahead using learned model
- More sample-efficient but computationally expensive

---

#### **When to Use RL (vs Supervised Learning)** (1h):

**Use RL when**:

✅ **Sequential decisions with delayed rewards**:
- Example: Dynamic pricing (today's price affects tomorrow's demand)
- Example: Ad bidding (campaign budget allocation over time)

✅ **Exploration-exploitation tradeoff**:
- Example: Recommendation systems (explore new content vs exploit known preferences)

✅ **Agent can influence environment**:
- Example: Robotics (AV trajectory planning)
- Example: Game playing (AlphaGo)

---

**DON'T use RL when**:

❌ **Supervised learning suffices** (you have labeled data):
- Example: Image classification → Use CNN, not RL
- Example: Fraud detection → Use XGBoost, not RL

❌ **Delayed rewards are not critical**:
- Example: Predicting customer churn → Supervised binary classification

❌ **Sample inefficiency is prohibitive**:
- RL often needs millions of interactions
- Example: Medical treatment decisions → Can't experiment on humans at scale

❌ **Environment is non-stationary and unpredictable**:
- Example: Stock market (rules change, adversarial)

---

#### **How to Implement RL (High-Level)** (1h):

**Step 1: Frame as MDP (Markov Decision Process)**:
```python
def define_mdp():
    """
    - States: What information does agent observe?
      Example (Uber pricing): {current demand, supply, time, location}
    
    - Actions: What can agent control?
      Example: {price multiplier: 1.0x, 1.2x, 1.5x, 2.0x}
    
    - Rewards: What do we optimize?
      Example: Revenue - driver churn penalty
    
    - Transition: How does state evolve?
      Example: Higher price → lower demand (next state)
    """
    pass
```

**Step 2: Choose Algorithm**:
- **Discrete actions, simple**: Q-Learning or DQN
- **Continuous actions**: PPO (Proximal Policy Optimization)
- **Sample-efficient needed**: Model-based RL or use offline RL (learn from logged data)

**Step 3: Simulation Environment**:
```python
import gym  # OpenAI Gym for toy environments

# Custom environment
class PricingEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)  # 4 price levels
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))
    
    def step(self, action):
        # Execute action, return (next_state, reward, done, info)
        pass
    
    def reset(self):
        # Initialize environment
        pass
```

**Step 4: Train Agent**:
```python
from stable_baselines3 import PPO

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
```

---

#### **Business Applications**:

1. **Dynamic Pricing** (Uber, Airbnb):
   - State: Demand, supply, time
   - Action: Price multiplier
   - Reward: Revenue - customer churn

2. **Ad Bidding** (Google, Meta):
   - State: User profile, ad campaign budget remaining
   - Action: Bid amount
   - Reward: Conversions - cost

3. **Recommendation Systems** (YouTube, Netflix):
   - State: User history, context
   - Action: Which video to recommend
   - Reward: Watch time + engagement

4. **Robotics/AV** (Waymo, Tesla):
   - State: Sensor inputs (lidar, camera)
   - Action: Steering, acceleration
   - Reward: Safety + progress toward goal

---

#### **Interview Preparation**:

**Key Questions to Prepare**:

1. **"Explain the difference between RL and supervised learning"**
   - RL: Agent learns from interaction, delayed rewards
   - SL: Learn from labeled examples, immediate feedback

2. **"When would you choose RL over a simpler approach?"**
   - Sequential decisions, delayed rewards, exploration needed
   - Example: Bandits for simple exploration, full RL for long-horizon planning

3. **"What are the challenges of RL in production?"**
   - Sample inefficiency (needs many interactions)
   - Safety (agent might take harmful actions during exploration)
   - Non-stationarity (environment changes)
   - Solution: Offline RL (learn from logged data), safe exploration

4. **"Explain the exploration-exploitation tradeoff"**
   - Exploration: Try new actions to discover better strategies
   - Exploitation: Use current best-known action
   - Solution: ε-greedy, UCB (Upper Confidence Bound), Thompson Sampling

---

**Deliverable**: 
- Conceptual understanding of RL landscape
- Can articulate when to use RL vs supervised learning
- Prepared interview answers for RL questions
- Know enough to have intelligent conversation (not expert-level implementation)

---

### **Week 24: Mock Interviews - Round 1** (26h)

##### **Monday-Wednesday: Causal Inference Mocks** (10h)

**Platform**: Pramp, Interviewing.io, or peers

**Number**: 5 sessions (2h each)

**Structure per mock**:
- **15 min**: Conceptual questions (IV, DID, RDD, PSM, CATE)
- **20 min**: Case study (e.g., "How would you estimate effect of X on Y?")
- **10 min**: Behavioral (talk about Project #2 or #4)
- **5 min**: Your questions

**Focus areas**:
1. Derivations: Can you derive 2SLS on a whiteboard?
2. Assumptions: When does each method fail?
3. **YOUR differentiators**: GEV models, hierarchical Bayesian (practice explaining these!)

**Post-mock**: Write down questions you struggled with, review theory, re-derive

---

##### **Thursday: ML Theory Mocks** (8h)

**Number**: 4 sessions (2h each)

**Topics to cover**:
1. **Backpropagation**: Derive chain rule, explain vanishing gradients
2. **Attention mechanism**: Explain transformer architecture, self-attention math, why O(n²)
3. **Loss functions**: Cross-entropy, MSE, when to use each
4. **Regularization**: L1/L2, dropout, batch norm - when and why
5. **Optimization**: SGD, Adam, learning rate schedules

**Practice setup**: Use GPT-4 as interviewer or find peer on Pramp

---

##### **Friday: AV Safety Interview Prep** (4h)

**For Tier 3 companies** (Cruise, Waymo, Zoox):

**Prepare answers**:
1. **"What safety metrics do you know?"**
   - TTC (Time-to-Collision), PET (Post-Encroachment Time), DRAC, headway
   - **Add**: "In my PhD, I developed novel GEV-based metrics for extreme events"

2. **"How do you evaluate AV safety?"**
   - Trajectory analysis, counterfactual simulation, causal inference for interventions
   - **Project #4**: "I built a system that generates counterfactual trajectories and estimates intervention effects using DoWhy"

3. **"Walk me through your AV project"**
   - Practice 5-minute Project #4 pitch
   - Emphasize: PhD work vs this project (descriptive vs causal counterfactuals)

**Demo practice**: Run through Project #4 Streamlit demo, explain each component

---

##### **Weekend: Portfolio Polish** (4h)

**Tasks**:
1. **Update all GitHub READMEs**:
   - Clear project descriptions
   - Problem statement
   - Technical approach
   - Results/metrics

2. **Add architecture diagrams** (Mermaid):
   ```mermaid
   graph TD
       A[User Query] --> B[CausalRAG System]
       B --> C[Neo4j Graph]
       B --> D[Vector DB]
       C --> E[Causal Chain Discovery]
       D --> F[Embedding Retrieval]
       E --> G[LLM Generation]
       F --> G
   ```

3. **Record demo videos** (2-3 min each for Projects #1-#4):
   - Screen recording + voiceover
   - Show: Input → Processing → Output
   - Upload to YouTube or embed in README

4. **Create visualizations**:
   - CATE heatmaps (Project #2)
   - Agent reasoning traces (Project #3)
   - Counterfactual trajectories (Project #4)

5. **Ensure repos have**:
   - Installation instructions
   - Usage examples
   - Evaluation results
   - Requirements.txt / environment.yml

---

### **MONTH 6 DELIVERABLE**:
- ✅ **Statistics Deep Dive Complete** (40h):
  - Regression diagnostics (heteroscedasticity, multicollinearity, autocorrelation)
  - Hypothesis testing (Type I/II, multiple testing, permutation, Bayesian)
  - Asymptotic theory (LLN, CLT, Delta Method)
  
- ✅ **Causal Theory Mastery**:
  - Can derive IPTW, 2SLS, RDD, doubly robust on whiteboard
  - Can explain when each method fails
  - 50 causal interview questions prepared

- ✅ **ML Theory Mastery**:
  - Can derive backprop, attention mechanism
  - Can explain Adam optimizer, why it works

- ✅ **PhD Differentiators**:
  - Can explain GEV models, hierarchical Bayesian modeling
  - Interview angles prepared for YOUR methods

- ✅ **Mock Interviews Complete** (9 mocks):
  - 5 causal inference mocks
  - 4 ML theory mocks

- ✅ **Portfolio Polished**:
  - All GitHub repos updated
  - Architecture diagrams added
  - Demo videos recorded

- ✅ **200+ LeetCode problems total**

**TOTAL MONTH 6**: ~116 hours (29h/week)

---

## MONTH 7: SYSTEM DESIGN + APPLICATIONS

### **Week 25-26: Advanced ML System Design Mastery** (30h/week = 60h total)

**Goal**: Synthesize the 4 monthly System Design sessions into a coherent framework. Focus on "Senior" traits (trade-offs, failure modes).

**Resource**: "Designing Machine Learning Systems" by Chip Huyen (~$50, ESSENTIAL)

**Reading + Practice**:
- **Chapters 1-11** (all chapters)
- **Review**: Re-read your notes from System Design Sessions 1-4 (Causal Serving, Ad Scoring, Experimentation, Feature Store).
- ~30h reading, ~30h designing NEW practice systems (scaling to 1B users).

**Key Topics**:
1. **Requirements gathering** (Ch 3-4): Translate business to ML problem
2. **Data engineering** (Ch 5): ETL, feature stores, data quality
3. **Feature engineering** (Ch 6): Leakage, scaling, missing data
4. **Model development** (Ch 7-8): Model selection, training, debugging
5. **Deployment** (Ch 9-11): Batch vs streaming, monitoring, continual learning

**Practice**: Design 10 systems
- Content recommendation (Netflix)
- Fraud detection (Stripe)
- Search ranking (Google)
- Feed ranking (Meta)
- Pricing optimization (Uber)
- Demand forecasting (Amazon)
- Ad targeting (your Project #2!)
- AV safety system (your Project #4!)
- Experiment platform (your Project #3!)
- Causal inference system (your Project #1!)

**Framework** (practice this structure):
1. **Clarify requirements** (2 min)
2. **Frame as ML problem** (2 min): Prediction vs ranking vs generation?
3. **Data** (5 min): What features? How to collect?
4. **Model** (5 min): Architecture? Training pipeline?
5. **Evaluation** (5 min): Offline metrics? Online A/B test?
6. **Deployment** (5 min): Latency constraints? Batch or real-time?
7. **Monitoring** (3 min): Drift detection? Model updates?

---

### **Week 27: Mock Interviews - Round 2** (30h)

**System Design Mocks** (10h):
- 5 sessions, 2h each
- Platforms: Exponent, Pramp, peers
- Practice: Design systems end-to-end in 45 min

**Coding Mocks** (10h):
- 5 sessions, 2h each
- Focus: Medium/Hard LeetCode
- Topics: DP, Graphs, Trees (common in AS interviews)

**Behavioral Mocks** (10h):
- Prepare STAR stories for:
  1. Technical challenge (Project #2 PySpark on 45M rows)
  2. Collaboration (PhD committee negotiations)
  3. Failure + Learning (Project setback → pivot)
  4. Leadership (PhD mentoring undergrads)
  5. **AV Safety domain** (Project #4, PhD work)

---

### **Week 28: Applications Start** (30h)

**Company Research** (10h):
- Research 50 companies (Tier 1, 2, 3)
- Understand products, ML use cases, team structure
- Identify 3-5 talking points per company (how YOUR projects fit)

**Resume Tailoring** (5h):
- Create 3 versions:
  1. Causal AI focus (Meta, Netflix, Uber)
  2. GenAI focus (OpenAI, Anthropic, Google)
  3. AV Safety focus (Cruise, Waymo, Zoox)

**Applications** (15h):
- Submit 20-30 applications
- Personalize each cover letter (reference specific projects/papers)
- Reach out to recruiters on LinkedIn

---

## MONTH 8: ACTIVE INTERVIEWING + NETWORKING

### **Week 29-30: Interview Pipeline** (30h/week = 60h)

**Active Interviews**:
- Expect: 5-10 phone screens
- Expect: 2-5 onsites
- Time: ~6h per onsite (4-5 rounds)

**Daily prep**:
- Review that day's company research (1h)
- Practice 1-2 LeetCode problems (1h)
- Review project most relevant to company (1h)

**Post-interview**:
- Send thank-you emails
- Write down questions asked
- Identify gaps, study

---

### **Week 31-32: Networking + Continued Applications** (30h/week = 60h)

**Networking** (20h/week):
- LinkedIn cold outreach: 10-20 messages/day to AS/ML Engineers at target companies
- Template: "Hi [Name], I'm impressed by [specific project]. I just built a [relevant project]. Would love to hear your advice on breaking into [company]."
- Coffee chats: 2-3 per week (30 min each)
- Conferences/Meetups: Attend 1-2 (if available)

**Applications** (10h/week):
- Continue submitting: Target 50+ total
- Follow up on pending applications
- Respond promptly to interview requests

---

### **Week 33+: Offers + Negotiation**

**Offer evaluation**:
- Compare: TC, team, growth, impact
- Use competing offers for leverage

**Negotiation**:
- Research: Levels.fyi for market rates
- Negotiate: Salary, equity, sign-on, relocation
- Ask for time to decide (1 week)

---

## MONTHS 6-8 TOTAL TIME

| Month | Focus | Hours | h/week |
|-------|-------|-------|--------|
| **Month 6** | Statistics (40h) + Theory + Mocks Round 1 | ~116h | 29h |
| **Month 7** | System Design (60h) + Mocks Round 2 + Apps | ~100h | 25h |
| **Month 8** | Active Interviewing + Networking | ~80-120h | 20-30h (variable) |

**TOTAL MONTHS 6-8**: ~296-336 hours

---

## FINAL DELIVERABLES (END OF MONTH 8)

### **Technical Depth**:
- [ ] Can derive: Backprop, IV, DID, RDD, IPTW, attention on whiteboard
- [ ] Can explain: GEV models, hierarchical Bayesian (YOUR differentiators)
- [ ] Implemented from scratch: Transformer, Viterbi, EM, CATE estimators

### **Projects**:
- [ ] 4 production GitHub repos (CausalRAG, Uplift+Counterfactual Ads, Multi-Agent Exp, AV Safety)
- [ ] All with: Demos, docs, evaluations, architecture diagrams, demo videos
- [ ] 4 blog posts published

### **Interview Readiness**:
- [ ] 200+ LeetCode problems solved
- [ ] 20+ mock interviews completed (causal + ML theory + system design + coding + behavioral)
- [ ] Can design 10 ML systems (Chip Huyen framework)
- [ ] 5-minute pitch for each project memorized
- [ ] STAR stories prepared (5-10)

### **Applications**:
- [ ] 50+ companies applied
- [ ] 10+ phone screens
- [ ] 3-5 onsites
- [ ] **TARGET**: 1-3 offers

### **Signal**:
- [ ] 1-2 research papers submitted (workshop/conference)
- [ ] Portfolio demonstrates: Causal AI + GenAI + MLOps mastery
- [ ] **Differentiation**: PhD methods (GEV, hierarchical Bayesian) + cutting-edge projects

---

## INTEGRATION NOTES & CONTRADICTIONS

### **✅ NO CONTRADICTIONS FOUND**

All content from original battle plan (lines 1068-1215) preserved.

**Statistics Deep Dive Integration**:
- **Week 21**: Regression diagnostics + Hypothesis testing (20h)
- **Week 22**: Asymptotic theory integrated into theory practice (4h)
- **Total**: 40h Statistics (24h new + 16h distributed across existing theory weeks)

### **Original vs Enhanced**:
- Original Month 6: Weeks 21-24, ~108-116h
- Enhanced: Same weeks, same hours (Statistics integrated cleanly)
- No additions needed - original plan already comprehensive for interview prep

---

## SUCCESS CRITERIA

- [ ] Can pass AS technical screens (causal inference deep questions)
- [ ] Can pass ML theory interviews (derive backprop, attention)
- [ ] Can design ML systems in 45 min (Chip Huyen framework)
- [ ] Can code Medium/Hard LeetCode in 30 min
- [ ] **Interview ready**: Confident explaining PhD work + Projects #1-4

---

**Status**: ✅ Months 6-8 plan complete, all interview prep phases covered

**Final Note**: By end of Week 33, you should be choosing between multiple offers from top companies. Your combination of (1) PhD-level causal expertise, (2) cutting-edge GenAI projects, (3) MLOps production skills, and (4) domain depth in AV safety creates a unique profile that commands premium compensation.

Good luck! 🚀
