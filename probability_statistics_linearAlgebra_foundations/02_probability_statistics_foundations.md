# Section 2: Mathematical & Statistical Foundations — Detailed Study Plan

---

## Document Outline
- [Executive Summary](#executive-summary)
- [2.1 Probability Basics](#21-probability-basics-h)
- [2.2 Distributions](#22-distributions-h)
- [2.3 Statistical Inference](#23-statistical-inference-h)
- [2.4 Advanced Statistical Theory](#24-advanced-statistical-theory-m)
- [2.5 Linear Algebra for ML](#25-linear-algebra-for-ml-h)
- [2.6 Calculus & Optimization](#26-calculus--optimization-h)
- [2.7 Information Theory](#27-information-theory-m)
- [2.8 Numerical Computation](#28-numerical-computation-l)
- [Deliverables & Verification](#deliverables--verification)
- [Time Allocation](#time-allocation)
- [Interview Question Bank](#interview-question-bank)

## Executive Summary
This document expands Section 2 of the Senior AS Study Agenda from 4 subsections (~40h) to 8 subsections (~55-60h). The original probability, distributions, inference, and advanced theory sections are enriched with interview-relevant sub-topics. Four new sections—Linear Algebra, Calculus & Optimization, Information Theory, and Numerical Computation—fill foundational gaps not covered elsewhere in the agenda (validated against Goodfellow's "Deep Learning" Part I). Depth is calibrated for senior AS interviews: heavy on intuition and applied derivations, light on measure-theoretic formalism.

---

> [!IMPORTANT]
> **Sections 2.5-2.8 are NEW additions** not in the original agenda. They cover foundational math that underpins ML/DL (Section 3), is assumed knowledge in senior interviews, and is not covered elsewhere. The time allocation in the main agenda (Phase 1, Weeks 3-7) should be adjusted accordingly—see [Time Allocation](#time-allocation).

> [!NOTE]
> **Cross-references**: ML Basics are covered in Section 3. This document focuses on the *mathematical machinery*; Section 3 covers the *application* of these tools to ML models. Where topics overlap (e.g., PCA uses SVD from 2.5; backprop uses chain rule from 2.6), this document covers the math and Section 3 covers the ML context.

---

## 2.1 Probability Basics **[H]**

> Build rock-solid probability intuition. Every statistics and ML concept builds on this.

### 2.1.1 Foundations of Probability **[H]**
- [ ] Sample spaces and events **[M]**
- [ ] Axioms of probability (Kolmogorov) **[M]**
- [ ] Conditional probability and independence **[H]**
- [ ] Law of Total Probability **[M]**
- [ ] Chain rule of probability **[H]**

**Interview tip**: Be ready to solve conditional probability puzzles (card draws, disease testing). These test intuition, not formulas.

### 2.1.2 Combinatorics **[M]**
- [ ] Permutations and combinations **[M]**
- [ ] Counting principles (multiplication, inclusion-exclusion) **[M]**

### 2.1.3 Random Variables & Expectations **[H]**
- [ ] Discrete vs continuous random variables **[H]**
- [ ] Probability Mass Function (PMF) and Probability Density Function (PDF) **[H]**
- [ ] Cumulative Distribution Function (CDF) **[H]**
- [ ] Expectation E[X]: definition, linearity, and applications **[C]**
- [ ] Variance Var(X) and Standard Deviation **[C]**
- [ ] Covariance and Correlation **[H]**
  - Cov(X,Y) = E[XY] - E[X]E[Y]
  - Correlation vs causation (critical interview point)
- [ ] Moment Generating Functions (MGF) **[L]**
  - Definition: M(t) = E[e^(tX)]
  - Know what they are; rarely asked directly in AS interviews

**Interview tip**: "Derive the variance of a Binomial" or "Show E[X^2] = Var(X) + (E[X])^2" are common whiteboard asks.

### 2.1.4 Bayes' Theorem **[C]**
- [ ] Intuition: updating beliefs with evidence **[C]**
- [ ] Formula derivation from conditional probability **[C]**
- [ ] Prior, Likelihood, Posterior, Evidence **[H]**
- [ ] Common interview applications **[C]**
  - Disease testing (sensitivity/specificity)
  - Spam filtering
  - A/B test interpretation
- [ ] Conjugate priors **[M]**
  - Beta-Binomial (most important)
  - Normal-Normal
  - Why conjugacy matters: closed-form posteriors

### 2.1.5 MLE vs MAP **[H]**
- [ ] Maximum Likelihood Estimation (MLE) derivation **[H]**
  - Log-likelihood trick
  - MLE for Normal, Bernoulli, Poisson
  - Properties: consistency, asymptotic normality, efficiency
- [ ] Maximum A Posteriori (MAP) derivation **[H]**
  - MAP = MLE + prior (regularization connection)
  - When MAP = MLE (uniform prior)
  - MAP with Gaussian prior = L2 regularization **[C]**
  - MAP with Laplace prior = L1 regularization **[C]**
- [ ] MLE vs MAP vs Full Bayesian: when to use each **[H]**

**Interview tip**: "Derive MLE for Normal distribution" and "Show that MAP with Gaussian prior gives Ridge regression" are classic whiteboard problems.

### 2.1.6 Probability Inequalities **[M]**
- [ ] Markov's inequality **[M]**
- [ ] Chebyshev's inequality **[M]**
- [ ] Jensen's inequality **[H]**
  - For convex f: f(E[X]) <= E[f(X)]
  - Key for: KL divergence non-negativity proof, EM algorithm derivation
- [ ] Hoeffding's inequality **[L]**
  - Application: bounding sample mean deviation
  - Connection to concentration inequalities in ML

**Why this matters**: These show up when discussing convergence guarantees, PAC learning bounds, and sample complexity arguments. Jensen's inequality is a frequent building block in ML theory proofs.

### Section 2.1 Deliverables
- [ ] Solve 10 conditional probability/Bayes problems without notes
- [ ] Derive MLE for Normal and Bernoulli on whiteboard
- [ ] Show MAP-regularization equivalence on whiteboard
- [ ] Derive Beta-Binomial conjugacy

---

## 2.2 Distributions **[H]**

> Know the shape, parameters, use cases, and relationships between distributions.

### 2.2.1 Discrete Distributions **[H]**
- [ ] Bernoulli: single trial, p **[H]**
- [ ] Binomial: n trials, successes **[H]**
  - Connection to Bernoulli (sum of iid Bernoullis)
- [ ] Poisson: rare events, rate lambda **[H]**
  - Poisson as limit of Binomial (n large, p small)
  - Applications: counts per time/area
- [ ] Geometric: trials until first success **[M]**
  - Memoryless property **[M]**
- [ ] Negative Binomial: trials until k-th success **[L]**
- [ ] Multinomial: multi-category extension of Binomial **[M]**
- [ ] When to use each distribution **[C]**

### 2.2.2 Continuous Distributions **[H]**
- [ ] Normal (Gaussian) **[C]**
  - Properties: symmetry, 68-95-99.7 rule
  - Standard Normal and Z-scores
  - Sum of Normals is Normal
- [ ] Exponential **[M]**
  - Memoryless property
  - Connection to Poisson process
- [ ] Uniform **[M]**
- [ ] Log-Normal **[M]**
  - When data is multiplicative (stock returns, income)
  - log(X) ~ Normal
- [ ] Beta distribution **[M]**
  - Prior for probabilities (Bayesian inference)
  - Shape parameters alpha, beta
- [ ] Gamma distribution **[L]**
  - Generalizes Exponential
  - Prior for rate parameters
- [ ] **Chi-squared distribution** **[H]** *(NEW)*
  - Sum of squared standard Normals
  - Degrees of freedom
  - Critical for hypothesis testing
- [ ] **t-distribution** **[H]** *(NEW)*
  - Heavy tails vs Normal
  - When to use (small samples, unknown variance)
  - Approaches Normal as df increases
- [ ] **F-distribution** **[M]** *(NEW)*
  - Ratio of two Chi-squared
  - ANOVA and regression F-tests
- [ ] **Dirac delta distribution** **[L]** *(Goodfellow 3.8)*
  - Used for empirical distributions
  - Conceptual: point mass at a single value
- [ ] Multivariate Normal **[H]** *(NEW)*
  - Mean vector and covariance matrix
  - Marginal and conditional distributions
  - Mahalanobis distance
  - Critical for: PCA, Gaussian processes, discriminant analysis

### 2.2.3 Key Theorems **[H]**
- [ ] Central Limit Theorem (CLT) **[C]**
  - Statement and intuition (sum of many iid -> Normal)
  - Conditions: finite mean and variance
  - Applications: confidence intervals, hypothesis tests
  - Know *why* it works (averaging reduces variance), skip formal proof
- [ ] Law of Large Numbers **[H]**
  - Weak LLN vs Strong LLN (convergence in probability vs a.s.)
  - Why it matters: justifies sample averages
- [ ] Delta Method **[M]**
  - Approximating distribution of g(X_bar)
  - Formula: Var(g(X)) ≈ [g'(mu)]^2 * Var(X)
  - Application: variance of ratio/product of estimators
  - Connection to A/B testing (ratio metrics)

### 2.2.4 Distribution Relationships **[M]** *(NEW)*
- [ ] Exponential family unification **[M]**
  - Natural parameter, sufficient statistic
  - Why important: GLMs, conjugate priors, MLE properties
- [ ] Key relationships diagram **[M]**
  - Bernoulli → Binomial → Normal (CLT)
  - Exponential → Gamma → Chi-squared
  - Poisson ↔ Exponential (arrival times)
  - Normal → Chi-squared → t → F

### Section 2.2 Deliverables
- [ ] Draw the distribution relationship diagram from memory
- [ ] For each distribution: write PMF/PDF, mean, variance, and one use case
- [ ] Sketch CLT convergence with code (increasing n)

---

## 2.3 Statistical Inference **[H]**

> The bridge from probability theory to data analysis. Core interview territory.

### 2.3.1 Estimation Theory **[H]**
- [ ] Point estimation **[H]**
  - Unbiasedness, consistency, efficiency **[H]**
  - Mean Squared Error = Bias^2 + Variance **[C]**
- [ ] Confidence intervals **[C]**
  - Interpretation: "95% of such intervals contain the true parameter"
  - Common misconception: "95% probability the parameter is in this interval" (WRONG for frequentist CI)
  - Z-intervals vs t-intervals (when to use each)
  - CI for proportions (Wald, Wilson)
- [ ] Standard error vs standard deviation **[H]**
  - SE = SD / sqrt(n)
  - SE of sample proportion = sqrt(p(1-p)/n)
- [ ] Bootstrap methods **[H]** *(NEW - expanded)*
  - Non-parametric bootstrap: resample with replacement
  - Parametric bootstrap: simulate from fitted model
  - Bootstrap confidence intervals (percentile, BCa)
  - When bootstrap fails (heavy tails, small n)

### 2.3.2 Hypothesis Testing **[C]**
- [ ] Framework **[C]**
  - Null and alternative hypotheses
  - One-sided vs two-sided tests
  - Test statistics
- [ ] p-values **[C]**
  - Correct interpretation: P(data this extreme | H0 true)
  - Why p < 0.05 is arbitrary
  - p-value is NOT P(H0 is true)
- [ ] Type I and Type II errors **[C]**
  - Alpha (significance level) controls Type I
  - Beta and Power (1 - Beta)
  - Tradeoff: reducing alpha increases beta
- [ ] Power analysis **[H]**
  - Effect size, sample size, alpha, power relationship
  - Minimum detectable effect (MDE)
  - Application: A/B test design
- [ ] Multiple testing corrections **[H]** *(NEW - expanded)*
  - Family-wise error rate (FWER)
  - Bonferroni correction (conservative)
  - False Discovery Rate (FDR) and Benjamini-Hochberg **[H]**
  - When to use FWER vs FDR control

### 2.3.3 Common Statistical Tests **[H]**
- [ ] t-tests **[H]**
  - One-sample, two-sample (independent), paired
  - Assumptions: normality, equal variance (Welch's for unequal)
  - When to use vs non-parametric alternatives
- [ ] Chi-square tests **[M]**
  - Goodness-of-fit
  - Test of independence
- [ ] ANOVA **[M]**
  - One-way ANOVA: comparing multiple group means
  - F-statistic derivation (between-group vs within-group variance)
  - Post-hoc tests (Tukey HSD) **[L]**
- [ ] Non-parametric tests **[M]** *(NEW)*
  - Mann-Whitney U (two-sample)
  - Wilcoxon signed-rank (paired)
  - Kruskal-Wallis (multi-group)
  - When to prefer over parametric tests
- [ ] Permutation / Randomization tests **[H]**
  - Exact test by shuffling labels
  - No distributional assumptions
  - Computationally intensive but powerful
  - Connection to causal inference (Fisher's sharp null)

### 2.3.4 Regression Foundations **[H]**
- [ ] OLS regression **[C]**
  - Derivation: minimize sum of squared residuals
  - Normal equations: beta = (X^T X)^(-1) X^T y
  - Gauss-Markov theorem (BLUE) **[M]**
  - Assumptions: linearity, independence, homoscedasticity, normality of errors
- [ ] Logistic regression **[C]**
  - Log-odds formulation
  - MLE estimation (no closed form)
  - Sigmoid function and decision boundary
- [ ] Generalized Linear Models (GLMs) **[M]** *(NEW)*
  - Link function concept
  - Poisson regression for count data
  - Connection to exponential families

### 2.3.5 Regression Diagnostics **[H]**
- [ ] Heteroscedasticity **[H]**
  - Visual detection (residual plots)
  - Breusch-Pagan test, White test
  - Robust standard errors (Huber-White)
- [ ] Multicollinearity **[H]**
  - Variance Inflation Factor (VIF)
  - Condition number
  - Consequences: unstable coefficients, inflated SE
- [ ] Autocorrelation **[M]**
  - Durbin-Watson test
  - Consequences in time series
  - Newey-West standard errors
- [ ] Influential points **[M]**
  - Leverage, Cook's distance
  - DFBETAS, DFFITS (conceptual)
- [ ] Residual analysis **[H]** *(NEW)*
  - Normal Q-Q plots
  - Residuals vs fitted values
  - Pattern recognition in residual plots

### Section 2.3 Deliverables
- [ ] Implement permutation test from scratch (Python, no libraries)
- [ ] Implement bootstrap CI from scratch
- [ ] Derive OLS normal equations on whiteboard
- [ ] Run full regression diagnostics on a dataset (code notebook)

---

## 2.4 Advanced Statistical Theory **[M]**

> Deeper theory for distinguished candidates. Separates good from great.

### 2.4.1 Asymptotic Theory **[M]**
- [ ] Key convergence concepts **[M]**
  - Convergence in probability (used in consistency)
  - Convergence in distribution (CLT)
  - Know the difference; skip formal measure-theoretic details
- [ ] Consistency of estimators **[M]**
  - Convergence in probability to true value
  - MSE consistency: Bias -> 0 and Var -> 0
- [ ] Asymptotic normality **[M]**
  - sqrt(n)(theta_hat - theta) -> N(0, V)
  - Why MLE is asymptotically Normal

### 2.4.2 Estimation Theory Deep Dive **[L-M]** *(NEW)*
- [ ] Sufficient statistics **[L]**
  - Conceptual: a statistic that captures all info about the parameter
  - Know the concept; skip factorization theorem proof
- [ ] Fisher Information **[M]**
  - Definition: I(theta) = -E[d^2/d(theta)^2 log f(x|theta)]
  - Role in MLE variance: Var(theta_hat) ~ 1/I(theta)
- [ ] Cramer-Rao Lower Bound **[L]**
  - Conceptual: no unbiased estimator can beat 1/I(theta) variance
  - Know the concept for interview discussion; skip derivation

### 2.4.3 Bayesian Foundations **[M]**
- [ ] Bayesian vs Frequentist philosophy **[M]**
  - Interpretation of probability
  - When each is appropriate
- [ ] MCMC intuition **[L]**
  - Metropolis-Hastings algorithm (conceptual)
  - Gibbs sampling (conceptual)
  - Why needed: intractable posteriors
- [ ] Bayes factors for model comparison **[L]**
- [ ] Credible intervals vs confidence intervals **[M]**

### 2.4.4 Causal & Statistical Connections **[M]** *(NEW)*
- [ ] Simpson's Paradox **[H]**
  - Why aggregated data can reverse conclusions
  - Resolution via conditioning (or not) on confounders
- [ ] Berkson's Paradox **[M]**
  - Collider bias
  - Selection bias examples
- [ ] Ecological Fallacy **[M]**
  - Group-level vs individual-level inference

### Section 2.4 Deliverables
- [ ] Compute Fisher Information for Bernoulli (simple exercise)
- [ ] Explain Simpson's Paradox with a concrete numerical example
- [ ] Articulate Bayesian vs Frequentist tradeoffs in 2 minutes

---

## 2.5 Linear Algebra for ML **[H]** *(NEW SECTION)*

> The language of ML. Every model manipulates vectors and matrices.

> [!NOTE]
> PCA application is in Section 3.2.2. This section covers the *math* that makes PCA work (eigendecomposition, SVD). Similarly, matrix calculus here supports backprop derivations in Section 3.3.1.

### 2.5.1 Vectors & Matrices **[H]**
- [ ] Vector operations: dot product, norms (L1, L2, L-inf) **[H]**
  - Geometric interpretation of dot product
  - Cosine similarity = normalized dot product
- [ ] Matrix operations: multiplication, transpose, inverse **[H]**
- [ ] **Determinant** **[M]** *(Goodfellow 2.11)*
  - Geometric meaning: volume scaling factor
  - Needed for: Gaussian PDF, eigenvalue computation (det(A - lambda I) = 0)
- [ ] **Trace** operator **[L]** *(Goodfellow 2.10)*
  - tr(A) = sum of diagonal elements = sum of eigenvalues
  - Shows up in ML derivations (e.g., matrix derivatives)
- [ ] Linear independence, rank, null space **[M]**
- [ ] Span and basis **[M]**
- [ ] Orthogonality and orthonormal bases **[M]**
- [ ] Projection onto subspaces **[H]**
  - proj_v(u) = (u . v / v . v) * v
  - Projection matrix: P = A(A^T A)^(-1) A^T
  - **Why it matters**: OLS is projection of y onto column space of X

### 2.5.2 Eigendecomposition **[H]**
- [ ] Eigenvalues and eigenvectors: Av = lambda v **[H]**
  - Intuition: directions that the matrix just scales
  - Characteristic polynomial: det(A - lambda I) = 0
- [ ] Eigendecomposition: A = Q Lambda Q^(-1) **[H]**
- [ ] Spectral theorem for symmetric matrices **[M]**
  - Real eigenvalues, orthogonal eigenvectors
  - A = Q Lambda Q^T
- [ ] Applications: PCA (top-k eigenvectors of covariance matrix) **[H]**

### 2.5.3 Singular Value Decomposition (SVD) **[H]**
- [ ] SVD: A = U Sigma V^T **[H]**
  - U: left singular vectors (column space)
  - Sigma: singular values (diagonal)
  - V: right singular vectors (row space)
- [ ] Low-rank approximation (Eckart-Young theorem) **[M]**
- [ ] Applications **[H]**
  - PCA via SVD (avoid computing covariance matrix explicitly)
  - Matrix completion (recommendations)
  - Pseudoinverse for least squares
- [ ] Truncated SVD for dimensionality reduction **[M]**

### 2.5.4 Positive Definite Matrices **[M]**
- [ ] Definition: x^T A x > 0 for all x != 0 **[M]**
- [ ] Properties: all positive eigenvalues, invertible **[M]**
- [ ] **Why it matters**: covariance matrices are PSD, Hessians at minima are PSD **[M]**

### 2.5.5 Matrix Calculus Essentials **[M]**
- [ ] Gradient of f(x): vector of partial derivatives **[H]**
- [ ] Jacobian matrix (vector-valued functions) **[M]**
- [ ] Hessian matrix (second derivatives) **[M]**
  - Positive definite Hessian → local minimum
- [ ] Common matrix derivatives **[M]**
  - d/dx (x^T a) = a
  - d/dx (x^T A x) = (A + A^T) x = 2Ax (if symmetric)
  - d/dx (||Ax - b||^2) = 2A^T(Ax - b) → leads to normal equations

### Section 2.5 Deliverables
- [ ] Derive OLS solution using matrix calculus
- [ ] Compute SVD by hand for a 2x2 matrix
- [ ] Implement PCA from scratch using eigendecomposition (NumPy only)
- [ ] Explain why covariance matrices are positive semi-definite

---

## 2.6 Calculus & Optimization **[H]** *(NEW SECTION)*

> Optimization is the engine of ML. Understand what gradient descent is actually doing.

### 2.6.1 Multivariate Calculus **[H]**
- [ ] Partial derivatives **[H]**
- [ ] Gradient: direction of steepest ascent **[H]**
- [ ] Directional derivatives **[M]**
- [ ] Chain rule (single and multivariate) **[C]**
  - Key for backpropagation derivation
  - Computational graph intuition
- [ ] Taylor series approximation **[M]**
  - First-order: linear approximation (gradient descent)
  - Second-order: quadratic approximation (Newton's method)
- [ ] **Useful functions** **[M]** *(Goodfellow 3.9)*
  - Sigmoid: sigma(x) = 1/(1+e^(-x)), properties, saturation
  - Softplus: zeta(x) = log(1 + e^x), smooth approx of ReLU
  - Relationship: sigma'(x) = sigma(x)(1-sigma(x))

### 2.6.2 Convexity **[H]**
- [ ] Convex sets and convex functions **[H]**
  - Definition: f(tx + (1-t)y) <= t f(x) + (1-t) f(y)
  - Visual intuition: "bowl-shaped"
- [ ] Why convexity matters for ML **[C]**
  - Convex → local minimum = global minimum
  - Linear/logistic regression loss is convex
  - Neural network loss is NOT convex
- [ ] Testing convexity: positive semi-definite Hessian **[M]**
- [ ] Concavity for log-likelihood maximization **[M]**

### 2.6.3 Unconstrained Optimization **[H]**
- [ ] First-order conditions: gradient = 0 **[H]**
- [ ] Second-order conditions: Hessian PSD/PD **[M]**
- [ ] Gradient Descent **[H]**
  - Update rule: theta = theta - alpha * gradient
  - Learning rate selection
  - Convergence conditions
- [ ] Stochastic Gradient Descent (SGD) **[H]**
  - Mini-batch gradient descent
  - Variance reduction techniques (conceptual)
- [ ] Momentum, Adam (intuition) **[M]**
  - Momentum: exponential moving average of gradients
  - Adam: adaptive learning rates per parameter
- [ ] Newton's Method **[M]**
  - Second-order: theta = theta - H^(-1) * gradient
  - Faster convergence but expensive (Hessian inversion)
  - Quasi-Newton methods (L-BFGS) conceptual

### 2.6.4 Constrained Optimization **[M]**
- [ ] Lagrange multipliers **[M]**
  - Equality constraints: L(x, lambda) = f(x) - lambda * g(x)
  - Application: derive PCA as constrained optimization
- [ ] KKT conditions (conceptual) **[L]**
  - Inequality constraints
  - Application: SVM dual formulation
- [ ] Duality (conceptual) **[L]**
  - Primal vs dual problems
  - Strong duality for convex problems

### 2.6.5 Integration Essentials **[M]** *(for completeness)*
- [ ] Integration as area / summation **[M]**
- [ ] PDF normalization: integral f(x)dx = 1 **[M]**
- [ ] Expectation as integral: E[X] = integral x f(x) dx **[H]**
- [ ] Change of variables **[M]**
- [ ] Monte Carlo integration (conceptual) **[L]**
  - Sample-based approximation of integrals
  - Foundation for MCMC methods

### Section 2.6 Deliverables
- [ ] Derive gradient descent update for linear regression from first principles
- [ ] Derive SVM margin maximization as constrained optimization (Lagrangian)
- [ ] Implement gradient descent from scratch for logistic regression
- [ ] Explain convexity → global optimum guarantee on whiteboard

---

## 2.7 Information Theory **[M]** *(NEW SECTION)*

> Foundational for loss functions, model selection, and drift detection.

### 2.7.1 Entropy **[H]**
- [ ] Shannon entropy: H(X) = -sum p(x) log p(x) **[H]**
  - Intuition: measure of uncertainty/surprise
  - Maximum entropy: uniform distribution
  - Minimum entropy: deterministic (all mass on one outcome)
- [ ] Differential entropy (continuous case) **[L]**
- [ ] Joint and conditional entropy **[M]**
  - H(X,Y) = H(X) + H(Y|X)
  - Chain rule of entropy

### 2.7.2 Cross-Entropy **[C]**
- [ ] Definition: H(p,q) = -sum p(x) log q(x) **[C]**
- [ ] **Why it's the standard classification loss** **[C]**
  - Cross-entropy loss = negative log-likelihood for categorical distributions
  - Binary cross-entropy: -[y log(p) + (1-y) log(1-p)]
  - Minimizing cross-entropy ≈ minimizing KL divergence from true distribution
- [ ] Connection to log-loss and MLE **[H]**

### 2.7.3 KL Divergence **[H]**
- [ ] Definition: KL(p||q) = sum p(x) log(p(x)/q(x)) **[H]**
- [ ] Properties **[H]**
  - Non-negative (Gibbs' inequality)
  - NOT symmetric: KL(p||q) != KL(q||p)
  - NOT a true distance metric
- [ ] Forward KL vs Reverse KL **[M]**
  - Forward KL (mean-seeking): used in variational inference
  - Reverse KL (mode-seeking): used in policy optimization
- [ ] Applications **[H]**
  - Data drift detection (PSI ≈ symmetrized KL)
  - Variational inference (ELBO)
  - Knowledge distillation

### 2.7.4 Mutual Information **[M]**
- [ ] Definition: I(X;Y) = KL(p(x,y) || p(x)p(y)) **[M]**
- [ ] I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) **[M]**
- [ ] Applications **[M]**
  - Feature selection (mutual information between feature and target)
  - Decision tree splitting (information gain = mutual information)
  - Independent component analysis

### Section 2.7 Deliverables
- [ ] Derive cross-entropy loss from MLE for logistic regression
- [ ] Show that KL divergence ≥ 0 (Jensen's inequality proof)
- [ ] Compute entropy and KL divergence by hand for simple distributions
- [ ] Explain the connection between decision tree information gain and mutual information

---

## 2.8 Numerical Computation **[L]** *(NEW SECTION)*

> Practical concerns that distinguish production-ready scientists from textbook ones.

### 2.8.1 Numerical Stability **[M]**
- [ ] Overflow and underflow **[M]**
  - Log-sum-exp trick for softmax **[H]**
  - Working in log-space for products of probabilities **[H]**

### 2.8.2 Conditioning **[M]**
- [ ] Condition number of a matrix **[M]**
  - kappa(A) = sigma_max / sigma_min
  - Ill-conditioned matrices → unstable solutions
- [ ] Relation to multicollinearity in regression **[M]**
- [ ] Regularization as a conditioning fix **[M]**

### 2.8.3 Practical Considerations **[M]**
- [ ] Numerical differentiation vs automatic differentiation **[M]**
  - Forward mode vs reverse mode autodiff
  - Why reverse mode is efficient for backprop (one backward pass)
- [ ] Numerical integration: Monte Carlo methods **[L]**
- [ ] Random number generation and seeds for reproducibility **[M]**

### Section 2.8 Deliverables
- [ ] Implement log-sum-exp and show numerical stability improvement
- [ ] Explain why adding regularization improves conditioning

---

## Deliverables & Verification

### Cumulative Deliverable Checklist

| # | Deliverable | Section | Priority |
|---|-------------|---------|----------|
| 1 | 10 Bayes/conditional probability problems (no notes) | 2.1 | H |
| 2 | Derive MLE for Normal and Bernoulli on whiteboard | 2.1 | H |
| 3 | Show MAP-regularization equivalence | 2.1 | H |
| 4 | Derive Beta-Binomial conjugacy | 2.1 | M |
| 5 | Distribution relationship diagram from memory | 2.2 | M |
| 6 | PMF/PDF, mean, variance, use case for each distribution | 2.2 | H |
| 7 | CLT convergence visualization (code) | 2.2 | M |
| 8 | Permutation test from scratch (Python) | 2.3 | H |
| 9 | Bootstrap CI from scratch (Python) | 2.3 | H |
| 10 | Derive OLS normal equations on whiteboard | 2.3 | C |
| 11 | Full regression diagnostics notebook | 2.3 | H |
| 12 | Compute Fisher Information for Bernoulli | 2.4 | M |
| 13 | Simpson's Paradox numerical example | 2.4 | H |
| 14 | OLS derivation via matrix calculus | 2.5 | H |
| 15 | SVD by hand (2x2) | 2.5 | M |
| 16 | PCA from scratch (eigendecomposition, NumPy) | 2.5 | H |
| 17 | Gradient descent for logistic regression from scratch | 2.6 | H |
| 18 | SVM Lagrangian derivation | 2.6 | M |
| 19 | Cross-entropy loss from MLE derivation | 2.7 | C |
| 20 | KL divergence >= 0 proof (Jensen's inequality) | 2.7 | M |
| 21 | Log-sum-exp stability demo | 2.8 | M |

### Verification Approach
- **Whiteboard tests**: Explain each derivation aloud as if in an interview
- **Code notebooks**: Each "from scratch" deliverable should be a standalone notebook
- **Spaced repetition**: Follow 1-7-30 schedule from main agenda for each sub-topic
- **Self-quiz**: After each subsection, close notes and answer 3 interview questions

---

## Time Allocation

### Original vs Expanded

| | Original | Expanded | Delta |
|---|----------|----------|-------|
| **Sections** | 2.1-2.4 | 2.1-2.8 | +4 sections |
| **Hours** | ~40h | ~55-60h | +15-20h |
| **Weeks** | 3-5 (3 weeks) | 3-7 (5 weeks) | +2 weeks |

### Proposed Weekly Schedule (Phase 1 Adjustment)

> [!WARNING]
> The original Phase 1 (Weeks 3-7) allocates 100h total to Stats + ML, with Weeks 3-5 for Stats and Weeks 4-7 interleaving with ML 3.1. The expansion below redistributes to give math foundations proper coverage while keeping ML on track.

| Week | Morning (Theory) | Afternoon (Practice) | Sections | Hours |
|------|------------------|---------------------|----------|-------|
| 3 | Prob Basics 2.1 (core probability, Bayes) | DSA + probability problems | 2.1 | 20h |
| 4 | Distributions 2.2 + Linear Algebra 2.5.1-2.5.2 | ML 3.1.1-3.1.2 | 2.2, 2.5, 3.1 | 20h |
| 5 | Inference 2.3 + Calculus 2.6.1-2.6.3 | ML 3.1.3-3.1.4 | 2.3, 2.6, 3.1 | 20h |
| 6 | Advanced Theory 2.4 + Info Theory 2.7 + Numerical 2.8 | ML 3.1.5-3.1.6 + Survival 9.3 | 2.4, 2.7, 2.8, 3.1, 9.3 | 20h |
| 7 | SVD/Optimization deep dive + Review | Mock Interview #1 (Stats/ML) | 2.5.3-2.5.5, 2.6.4, Review | 20h |

### Per-Section Hour Estimates

| Section | Hours | Strategy |
|---------|-------|----------|
| 2.1 Probability Basics | 8-10h | Deep: many interview problems |
| 2.2 Distributions | 5-7h | Study cards + relationships |
| 2.3 Statistical Inference | 10-12h | Deep: coding + theory |
| 2.4 Advanced Statistical Theory | 4-5h | Conceptual: know the ideas, skip heavy proofs |
| 2.5 Linear Algebra for ML | 8-10h | Focused: eigen, SVD, matrix calc |
| 2.6 Calculus & Optimization | 8-10h | Focused: gradient descent math |
| 2.7 Information Theory | 4-5h | Moderate: loss function connections |
| 2.8 Numerical Computation | 2-3h | Light: practical tips only |
| **Total** | **~55-60h** | |

---

## Interview Question Bank

### Probability (2.1)
1. "A disease affects 1 in 1000 people. A test has 99% sensitivity and 95% specificity. If someone tests positive, what's the probability they have the disease?"
2. "Derive MLE for the parameter p of a Bernoulli distribution."
3. "How does MAP relate to regularization? Show the connection."
4. "What's the difference between MLE and MAP? When would you prefer one over the other?"

### Distributions (2.2)
5. "When would you use a Poisson vs Binomial distribution?"
6. "Explain CLT in your own words. Why is it important for A/B testing?"
7. "What is the Delta Method and when would you use it?"
8. "Describe the multivariate Normal. What does the covariance matrix tell you?"

### Inference (2.3)
9. "Your A/B test shows p = 0.04. What does this mean? What does it NOT mean?"
10. "You're running 20 simultaneous tests. How do you control for multiple comparisons?"
11. "Walk me through designing an A/B test—how do you choose sample size?"
12. "Your regression shows high R-squared but VIF > 10. What's happening and how do you fix it?"
13. "What are the assumptions of OLS regression? What happens if each is violated?"

### Advanced Theory (2.4)
14. "What is the Cramer-Rao bound and why does it matter?"
15. "Explain Simpson's Paradox with an example."
16. "What's the difference between a confidence interval and a credible interval?"

### Linear Algebra (2.5)
17. "Explain PCA in terms of eigenvalues and eigenvectors."
18. "What is SVD and how is it used in recommendation systems?"
19. "Derive the OLS solution beta = (X^T X)^(-1) X^T y using matrix calculus."
20. "What does it mean for a covariance matrix to be positive semi-definite?"

### Optimization (2.6)
21. "Why does gradient descent work? What conditions guarantee convergence?"
22. "What is convexity and why does it simplify optimization?"
23. "Explain the difference between gradient descent, SGD, and Adam."
24. "Derive the Lagrangian for SVM margin maximization."

### Information Theory (2.7)
25. "Why do we use cross-entropy loss instead of MSE for classification?"
26. "What is KL divergence? Why is it not symmetric?"
27. "How is information gain in decision trees related to mutual information?"

### Numerical Computation (2.8)
28. "What is the log-sum-exp trick and why is it needed?"
29. "What makes a matrix ill-conditioned and how does regularization help?"

---

> [!TIP]
> **Study order recommendation**: Start with 2.1 → 2.2 → 2.5 (linear algebra) → 2.6 (calculus) → 2.3 → 2.7 → 2.4 → 2.8. This builds math foundations (2.5, 2.6) before inference (2.3) and information theory (2.7), which rely on them.
