---
title: "2.2 Distributions"
---

> [!IMPORTANT]
> ## How to Study This File (Read This First)
>
> **Why distributions matter**: Every ML model either *assumes* a distribution (linear regression assumes Normal errors; logistic regression assumes Bernoulli outputs; Poisson regression assumes count data) or *outputs* one (a softmax layer outputs a Categorical). Every loss function is a negative log-likelihood of some distribution. Every hypothesis test is built on Chi-squared, t, or F. If you don't know distributions, you don't know *why* any of it works — you just know API calls.
>
> **The payoff**: After this file, Section 2.3 (Statistical Inference) will make immediate sense because you'll already know *why* t-tests use the t-distribution, *why* ANOVA uses F, and *why* sample proportions are approximately Normal. Causal inference (Section 1.2) will click too — A/B test power analysis is just Binomial → Normal. Bayesian A/B testing (Thompson Sampling) is literally Beta distributions. The material here unlocks everything downstream.
>
> ---
>
> ### Suggested Study Plan (5–7 hours across 2 sessions)
>
> **Session 1 (~3h): Discrete + Decision Guides**
> 1. Read [2.2.1 Discrete Distributions](#221-discrete-distributions-h) — one distribution at a time, don't rush
> 2. After each: close the file and write from memory: *PMF, mean, variance, one real use case*
> 3. Do the [Which Discrete Distribution?](#which-discrete-distribution-decision-guide) flowchart — trace through it with 3 real examples you invent
> 4. Review the [Discrete Summary Table](#discrete-distributions-summary-table) — this is your recall test
>
> **Session 2 (~3–4h): Continuous + Theorems + Relationships**
> 1. Read [2.2.2 Continuous Distributions](#222-continuous-distributions-h) — prioritize **Normal, Beta, Chi-squared, t** (all "Must know")
> 2. Do the [Which Continuous Distribution?](#which-continuous-distribution-decision-guide) flowchart with your own examples
> 3. Read [2.2.3 Key Theorems](#223-key-theorems-h) — CLT and LLN are [C] Critical; don't skip
> 4. Skim [2.2.4 Distribution Relationships](#224-distribution-relationships-m) — the diagram is the key artifact; sketch it from memory after
>
> ---
>
> ### Priority Triage (if time is tight)
>
> | Must master before moving on | Can skim now, revisit later |
> |---|---|
> | Bernoulli, Binomial, Poisson, Normal | Geometric, Negative Binomial |
> | Beta (Bayesian A/B testing) | Gamma, Dirac Delta |
> | Chi-squared, t-distribution | F-distribution |
> | CLT (full intuition + applications) | Exponential Family (2.2.4) |
> | Multivariate Normal | Delta Method details |
>
> ---
>
> ### Active Recall Protocol (do this, don't just read)
>
> For each distribution, before turning the page, ask yourself:
> - What does $X$ represent in the real world?
> - What are the parameters and what do they control?
> - What are the mean and variance? (derive, don't look up)
> - When would I choose this over the alternatives?
> - What ML concept directly uses this distribution?
>
> **Spaced repetition**: On Day 7, re-do the two flowcharts and the summary tables from memory. That's your retention check.

---
# Document Outline
- [Executive Summary](#executive-summary)
- [2.2.1 Discrete Distributions](#221-discrete-distributions-h)
  - [Foundational Concepts: Support, Expectation, and Variance](#foundational-concepts-support-expectation-and-variance)
  - [Bernoulli Distribution](#bernoulli-distribution)
  - [Binomial Distribution](#binomial-distribution)
  - [Poisson Distribution](#poisson-distribution)
  - [Geometric Distribution](#geometric-distribution)
  - [Negative Binomial Distribution](#negative-binomial-distribution)
  - [Multinomial Distribution](#multinomial-distribution)
  - [Which Discrete Distribution?](#which-discrete-distribution-decision-guide)
  - [Discrete Summary Table](#discrete-distributions-summary-table)
- [2.2.2 Continuous Distributions](#222-continuous-distributions-h)
  - [Normal (Gaussian)](#normal-gaussian-distribution)
  - [Exponential](#exponential-distribution)
  - [Uniform](#uniform-distribution)
  - [Log-Normal](#log-normal-distribution)
  - [Beta](#beta-distribution)
  - [Gamma](#gamma-distribution)
  - [Chi-Squared](#chi-squared-distribution)
  - [t-Distribution](#t-distribution)
  - [F-Distribution](#f-distribution)
  - [Dirac Delta](#dirac-delta-distribution)
  - [Multivariate Normal](#multivariate-normal-distribution)
  - [Which Continuous Distribution?](#which-continuous-distribution-decision-guide)
  - [Continuous Summary Table](#continuous-distributions-summary-table)
- [2.2.3 Key Theorems](#223-key-theorems-h)
  - [Central Limit Theorem](#central-limit-theorem-clt)
  - [Law of Large Numbers](#law-of-large-numbers)
  - [Delta Method](#delta-method)
- [2.2.4 Distribution Relationships](#224-distribution-relationships-m)
  - [Exponential Family](#exponential-family-unification)
  - [Relationship Diagram](#key-relationships-diagram)
- [Connections Map](#connections-map)
- [Interview Cheat Sheet](#interview-cheat-sheet)
- [Learning Objectives Checklist](#learning-objectives-checklist)

# Executive Summary

This guide covers Section 2.2: Distributions — the vocabulary of randomness. Every ML model assumes or outputs a probability distribution; knowing the shape, parameters, use cases, and relationships between distributions is essential for model selection, loss function design, and statistical testing. Content is calibrated against Goodfellow et al. *Deep Learning* (Chapter 3) and structured for senior Applied Scientist interview preparation. Each distribution progresses from definition to intuition to properties to Python visualization to ML connections.

> **Primary Reference**: Goodfellow, I., Bengio, Y., and Courville, A. *Deep Learning*. MIT Press, 2016.
> Chapter 3: Probability and Information Theory (pp. 53-76).

### Goodfellow Cross-Reference Map

Use this to read alongside your physical copy:

| This Guide | Goodfellow Section | Book Pages | What to Read |
|---|---|---|---|
| **2.2.1** Bernoulli/Multinomial | 3.9.1 Bernoulli | p. 63 | Bernoulli definition |
| **2.2.1** Multinomial | 3.9.2 Multinoulli (Categorical) | p. 63 | Single-trial multinomial |
| **2.2.2** Normal | 3.9.3 Gaussian Distribution | pp. 63-65 | Normal PDF, precision, CLT motivation |
| **2.2.2** Exponential/Laplace | 3.9.4 Exponential and Laplace | p. 65 | Sharp peak at 0, L1 connection |
| **2.2.2** Dirac delta | 3.9.5 Dirac Distribution | p. 65 | Empirical distribution definition |
| **2.2.2** Multivariate Normal | 3.9.3 Gaussian Distribution | pp. 63-65 | Covariance matrix, precision matrix |
| **2.2.3** CLT | 3.9.3 (motivation) | p. 64 | Why Normal is "default" — CLT justification |
| **2.2.4** Exponential family | — | — | Not covered directly; see Bishop Ch. 2 |
| **2.2.2** Sigmoid/Softplus | 3.10 Useful Properties | pp. 65-66 | Sigmoid, softplus, and their relationships |

> [!TIP]
> **Reading strategy**: Goodfellow Chapter 3.9 is a quick reference for key distributions — it's intentionally brief. Use this guide for depth and intuition, then glance at Goodfellow for the canonical definitions. For the testing distributions (Chi-squared, t, F), use any standard statistics textbook.

---

# 2.2 Distributions

> **Study Time**: 5-7 hours | **Priority**: [H] High | **Goal**: Know the shape, parameters, use cases, and relationships between distributions.

---

## 2.2.1 Discrete Distributions **[H]**

> **Book**: Goodfellow Ch. 3.9.1-3.9.2 (p. 63) | Bernoulli and Multinoulli (Categorical)

> Every classification model outputs a discrete distribution. Understanding these distributions means understanding what your model is actually predicting.

---

### Foundational Concepts: Support, Expectation, and Variance

Before diving into individual distributions, you need to understand **three concepts** that appear in every single distribution's property table. This section explains *what* they are, *why* they matter, and *how* the math derives them — first in general, then worked through step-by-step with the Bernoulli.

---

#### What Does "Support" Mean?

**Plain English**: The **support** of a distribution is the set of values that can actually happen — the values where the probability is not zero.

Think of it as the "menu" of outcomes the random variable is allowed to produce.

| Distribution | Support | In Plain English |
|---|---|---|
| Bernoulli | $\{0, 1\}$ | Only two outcomes: success or failure |
| Binomial$(n, p)$ | $\{0, 1, 2, \ldots, n\}$ | You can get anywhere from 0 to $n$ successes |
| Poisson | $\{0, 1, 2, \ldots\}$ | Any non-negative count (no upper limit) |
| Normal | $(-\infty, +\infty)$ | Any real number, in theory |
| Exponential | $[0, +\infty)$ | Any non-negative real number (waiting time can't be negative) |

**Why it matters**:
- Support tells you **where to sum** (discrete) or **where to integrate** (continuous) when computing expectations, variances, probabilities, and likelihoods.
- It constrains what a model can predict. A Poisson regression can't predict negative counts — because $-3$ is not in the support. A Normal regression *can* predict negative values — because $(-\infty, +\infty)$ is its support.
- When you write software to simulate from or evaluate a distribution, you need to know the valid domain.

---

#### How to Derive Expectation (Mean) — General Framework

**Plain English**: The expectation $E[X]$ is the **long-run average** of a random variable. If you repeated the experiment millions of times and averaged all the results, the answer would converge to $E[X]$.

##### Discrete Case

$$E[X] = \sum_{x \in \text{support}} x \cdot P(X = x)$$

**What this formula says in words**: "Go through every possible outcome $x$. For each one, ask: how likely is it [$P(X = x)$]? Weight the outcome by its probability. Add everything up."

**Why this works**: Outcomes that happen often get heavy weight; outcomes that rarely happen get near-zero weight. The weighted sum gives you the "center of mass" — the value the distribution leans toward over the long run.

**Analogy**: Imagine a weighted die. To find its average roll, you wouldn't just compute $(1+2+3+4+5+6)/6$ — you'd weight each face by how often it actually appears. That's exactly what $E[X]$ does.

##### Continuous Case

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

Same idea — but instead of summing over discrete outcomes, you integrate over all real values using the probability density function $f(x)$. The integral limits are effectively the support (since $f(x) = 0$ outside it).

---

#### How to Derive Variance — General Framework

**Plain English**: The variance $\text{Var}(X)$ measures **how spread out** the outcomes are around the mean. A small variance means outcomes cluster tightly around $E[X]$; a large variance means they're scattered.

##### The Definition

$$\text{Var}(X) = E\big[(X - \mu)^2\big]$$

where $\mu = E[X]$.

**What this says in words**: "For each possible outcome, compute how far it is from the mean $(X - \mu)$. Square that distance (so negatives don't cancel positives). Then take the expected value (probability-weighted average) of those squared distances."

##### The Shortcut Formula

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

**Why this shortcut exists**: Expanding $(X - \mu)^2 = X^2 - 2\mu X + \mu^2$ and taking expectations gives $E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$. This is often *much* easier to compute than the definition, because you only need $E[X]$ and $E[X^2]$.

##### Discrete Case

$$\text{Var}(X) = \sum_{x \in \text{support}} (x - \mu)^2 \cdot P(X = x) \quad \text{or equivalently} \quad \sum_{x} x^2 \cdot P(X = x) - \mu^2$$

##### Continuous Case

$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot f(x) \, dx \quad \text{or equivalently} \quad \int_{-\infty}^{\infty} x^2 \cdot f(x) \, dx - \mu^2$$

---

#### Worked Derivation: Bernoulli Expectation and Variance

Now let's apply the general framework to the simplest distribution.

**Setup**: $X \sim \text{Bernoulli}(p)$, meaning $X = 1$ with probability $p$, and $X = 0$ with probability $1-p$.

**Support**: $\{0, 1\}$ — so every sum has exactly two terms.

##### Deriving $E[X]$

$$E[X] = \sum_{x \in \{0, 1\}} x \cdot P(X = x)$$

Expand the sum — there are only two values in the support:

$$E[X] = 0 \cdot P(X = 0) + 1 \cdot P(X = 1)$$

$$= 0 \cdot (1 - p) + 1 \cdot p$$

$$= p$$

**Plain English**: The first term is "the value 0, weighted by its probability $(1-p)$" — but $0 \times$ anything $= 0$, so failures contribute nothing to the average. The second term is "the value 1, weighted by its probability $p$" — which is just $p$. So the long-run average of a 0/1 coin flip is simply the probability of success. This should feel intuitive: if a coin lands heads 30% of the time, the average of millions of flips is 0.30.

##### Deriving $\text{Var}(X)$ (using the shortcut)

**Step 1**: Compute $E[X^2]$

$$E[X^2] = \sum_{x \in \{0, 1\}} x^2 \cdot P(X = x) = 0^2 \cdot (1-p) + 1^2 \cdot p = p$$

Wait — $E[X^2] = p$ — the same as $E[X]$? Yes! Because $0^2 = 0$ and $1^2 = 1$, so squaring 0s and 1s changes nothing.

**Step 2**: Apply the shortcut

$$\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$$

**Plain English**: The variance is $p(1-p)$. This is a product of "how likely success is" times "how likely failure is." Think about the extremes:
- If $p = 0$ (never succeeds) or $p = 1$ (always succeeds), there's **no uncertainty** — you know the outcome — so variance $= 0$.
- If $p = 0.5$ (pure coin flip), uncertainty is **maximized** — you genuinely can't predict the outcome — so variance $= 0.25$, the highest possible.

The formula $p(1-p)$ naturally captures this: it's a downward parabola that peaks at $p = 0.5$ and hits zero at the endpoints.

> [!TIP]
> **Derivation template**: For every distribution in this file, the expectation and variance are derived using the exact same recipe:
> 1. Write out the definition: $E[X] = \sum x \cdot P(X=x)$ (or the integral for continuous)
> 2. Substitute the PMF/PDF formula
> 3. Simplify using algebra, known series, or the shortcut $\text{Var} = E[X^2] - (E[X])^2$
>
> The only thing that changes between distributions is the PMF/PDF and the support. The *method* is always the same.

---

### Bernoulli Distribution

The simplest possible distribution: a single trial with two outcomes.

**Why learn this**: Every binary classifier you build (logistic regression, neural net with sigmoid) models its output as Bernoulli. When you compute binary cross-entropy loss, you're computing the negative log of this PMF. You'll use this every single day.

**Used directly in**:
- **Binary cross-entropy loss**: `torch.nn.BCELoss()` and `sklearn.metrics.log_loss()` are the negative log of this PMF — you compute Bernoulli likelihood every time you train a binary classifier
- **Dropout implementation**: generating the dropout mask is literally `torch.bernoulli(torch.full(shape, p))` — each neuron is an independent Bernoulli trial
- **Conversion rate modeling**: when estimating P(purchase | visit), you model each visit as Bernoulli(p) and use the variance formula p(1-p) to compute standard errors for A/B tests

$$X \sim \text{Bernoulli}(p)$$

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

**Symbol definitions**:
- $X$ = the random variable: the outcome of a single trial (0 = failure, 1 = success)
- $p$ = **parameter** you set: the probability of success on that trial
- $x$ = **query value** you plug in: either 0 or 1

**What it outputs**: You give it $x = 0$ or $x = 1$, and it returns the probability of that outcome. That's it — it answers "how likely is success?" ($p$) or "how likely is failure?" ($1-p$).

| Property | Value |
|----------|-------|
| **Parameters** | $p \in [0, 1]$ (success probability) |
| **Support** | $\{0, 1\}$ |
| **Mean** | $E[X] = p$ |
| **Variance** | $\text{Var}(X) = p(1-p)$ |
| **Maximum variance** | At $p = 0.5$ (maximum uncertainty) |

> [!NOTE]
> **Variance intuition**: Variance $p(1-p)$ is maximized at $p=0.5$ and equals zero at $p=0$ or $p=1$. This is why A/B tests with 50/50 conversion rates need the largest sample sizes — you have maximum uncertainty about the outcome.

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Binary Classification Output (Agenda 3.1):**
> Every binary classifier (logistic regression, neural network with sigmoid output) models the target as a Bernoulli random variable. The model predicts $\hat{p} = P(Y=1 \mid X)$, and the loss function (binary cross-entropy) is derived directly from the Bernoulli likelihood:
> $$\mathcal{L} = -[y \log(\hat{p}) + (1-y) \log(1-\hat{p})]$$
> This is literally the negative log of the Bernoulli PMF! Minimizing cross-entropy = maximizing Bernoulli likelihood.
>
> **2. Dropout Regularization (Agenda 3.3.2):**
> Each neuron in dropout is independently "kept" with probability $p$ — a Bernoulli trial. The dropout mask for a layer of $n$ neurons is a vector of $n$ independent Bernoulli random variables.

---

### Binomial Distribution

The number of successes in $n$ independent Bernoulli trials.

**Why learn this**: When your PM asks "how many conversions should we expect from 10,000 visitors?" — that's Binomial. It's the exact distribution behind every A/B test sample size calculator you'll ever use.

**Used directly in**:
- **Exact binomial test**: `scipy.stats.binomtest(k, n, p)` — use when sample is too small for z-test approximation (e.g., testing 8 conversions out of 50 visitors)
- **A/B test sample size calculators**: the SE formula $\sqrt{p(1-p)/n}$ comes from Binomial variance divided by $n^2$ — every sample size calculator you'll build starts here
- **Sequential testing boundaries**: group sequential designs use the Binomial distribution to set stopping boundaries as data accumulates

$$X \sim \text{Binomial}(n, p)$$

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

**Symbol definitions**:
- $X$ = the random variable: the total number of successes out of $n$ trials
- $n$ = **parameter** you set: how many independent trials you run
- $p$ = **parameter** you set: the probability of success on each trial
- $k$ = **query value** you plug in: "what if exactly this many succeed?"
- $\binom{n}{k}$ = the number of ways to arrange $k$ successes among $n$ trials ("$n$ choose $k$")

**What it outputs**: You give it a count $k$, and it returns the probability of getting exactly $k$ successes out of $n$ trials. Example: "What's the probability that exactly 60 out of 1000 visitors convert?"

| Property | Value |
|----------|-------|
| **Parameters** | $n \in \mathbb{N}$ (trials), $p \in [0, 1]$ (success probability) |
| **Support** | $\{0, 1, 2, \ldots, n\}$ |
| **Mean** | $E[X] = np$ |
| **Variance** | $\text{Var}(X) = np(1-p)$ |
| **Connection** | Sum of $n$ i.i.d. Bernoulli$(p)$ |

> [!IMPORTANT]
> **Key relationship**: If $X_1, X_2, \ldots, X_n \sim \text{Bernoulli}(p)$ are i.i.d., then $\sum_{i=1}^n X_i \sim \text{Binomial}(n, p)$. This is your first example of a distribution arising from summing simpler ones — a pattern that repeats throughout this section.

<details>
<summary><strong>Worked Example: A/B Test Counts</strong></summary>

**Setup**: You run an A/B test with $n = 1000$ visitors. The current conversion rate is $p = 0.05$ (5%).

**Question**: What is the probability of observing exactly 60 conversions?

$$P(X = 60) = \binom{1000}{60} (0.05)^{60} (0.95)^{940}$$

```python
from scipy import stats

n, p = 1000, 0.05
# Exact probability of exactly 60 conversions
print(f"P(X=60) = {stats.binom.pmf(60, n, p):.6f}")  # 0.014...

# More useful: P(X >= 60) — "is 60 conversions unusually high?"
print(f"P(X>=60) = {1 - stats.binom.cdf(59, n, p):.4f}")  # ~0.095

# Normal approximation (CLT preview): works when np >= 5 and n(1-p) >= 5
mu, sigma = n*p, (n*p*(1-p))**0.5
print(f"Normal approx: mu={mu:.1f}, sigma={sigma:.2f}")   # mu=50, sigma=6.89
```

**Insight**: With $n=1000$ and $p=0.05$, the expected count is $np=50$ with $\sigma \approx 6.9$. Seeing 60 conversions is about 1.45 standard deviations above the mean — notable but not statistically significant at $\alpha = 0.05$ (it's within the 90% range).

</details>

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. A/B Testing Sample Size (Agenda 1.2.1):**
> The Binomial distribution is the exact model for "how many users convert out of $n$ visitors." The Normal approximation to the Binomial (valid when $np \geq 5$ and $n(1-p) \geq 5$) is what makes z-test based A/B testing possible. When you calculate the standard error $\text{SE} = \sqrt{p(1-p)/n}$, you are using the Binomial variance $np(1-p)$ divided by $n^2$.
>
> **2. Binomial Likelihood and Beta-Binomial Conjugacy (Section 2.1.4):**
> In Bayesian A/B testing, the Binomial is the *likelihood* and the Beta distribution is the conjugate *prior*. After observing $k$ successes in $n$ trials, the posterior is $\text{Beta}(\alpha + k, \beta + n - k)$ — a closed-form update.

---

### Poisson Distribution

Models the count of events occurring in a fixed interval (time, area, volume) at a constant average rate.

**Why learn this**: At Amazon, daily order counts per warehouse, server errors per hour, and customer support tickets per shift are all Poisson. When you build a demand forecasting model or anomaly detector for count data, Poisson (or its overdispersed cousin, Negative Binomial) is your starting point.

**Used directly in**:
- **Poisson regression**: `sm.GLM(y, X, family=sm.families.Poisson())` — the standard model for count data (daily orders, support tickets, bug reports)
- **Poisson loss in gradient boosting**: `XGBRegressor(objective='count:poisson')` and `LGBMRegressor(objective='poisson')` — use when predicting counts instead of MSE
- **Anomaly detection for counts**: if your server normally gets $\lambda=5$ errors/hour, observing 15 has p-value `1 - scipy.stats.poisson.cdf(14, 5)` $\approx 0.0003$ — that's an alert
- **Rate comparison**: testing whether two Poisson rates differ (e.g., defect rates between factories) uses the conditional exact test or E-test

$$X \sim \text{Poisson}(\lambda)$$

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

**Symbol definitions**:
- $X$ = the random variable: the count of events that occur in a **fixed interval** (the interval is decided by you, e.g., "per hour")
- $\lambda$ = **parameter** you set: the average rate of events per interval (e.g., 3 errors/hour)
- $k$ = **query value** you plug in: "what if exactly this many events happen?"
- $e^{-\lambda}$ = baseline probability factor that ensures everything sums to 1
- $k!$ = factorial, accounts for the ordering of events

**What it outputs**: You give it a count $k$, and it returns the probability of seeing exactly $k$ events in your fixed interval. The interval itself is baked into $\lambda$ — the distribution does not output a time or interval, only a probability for a given count.

| Property | Value |
|----------|-------|
| **Parameters** | $\lambda > 0$ (rate / expected count) |
| **Support** | $\{0, 1, 2, \ldots\}$ (unbounded) |
| **Mean** | $E[X] = \lambda$ |
| **Variance** | $\text{Var}(X) = \lambda$ |
| **Key property** | Mean = Variance (equidispersion) |

> [!NOTE]
> **Mean = Variance** is the defining fingerprint of a Poisson. If your count data has variance much larger than its mean, it is **overdispersed** and Negative Binomial is a better fit. This is one of the most common modeling mistakes in practice.

**Poisson as a limit of Binomial**: When $n$ is large and $p$ is small (but $\lambda = np$ is moderate):

$$\text{Binomial}(n, p) \approx \text{Poisson}(np)$$

**Rule of thumb**: Use this approximation when $n \geq 20$ and $p \leq 0.05$.

<details>
<summary><strong>Worked Example: Website Errors</strong></summary>

**Setup**: A server averages $\lambda = 3$ errors per hour.

**Question 1**: What is the probability of zero errors in the next hour?

$$P(X = 0) = \frac{3^0 e^{-3}}{0!} = e^{-3} \approx 0.0498$$

**Question 2**: What is the probability of more than 5 errors?

$$P(X > 5) = 1 - P(X \leq 5) = 1 - \sum_{k=0}^{5} \frac{3^k e^{-3}}{k!}$$

```python
from scipy import stats

lam = 3
print(f"P(X=0) = {stats.poisson.pmf(0, lam):.4f}")       # 0.0498
print(f"P(X>5)  = {1 - stats.poisson.cdf(5, lam):.4f}")   # 0.0839
```

</details>

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Poisson Regression / GLMs (Agenda 2.3.4):**
> When modeling count data (page views, clicks, defect counts), Poisson regression uses a log link: $\log(\lambda) = X\beta$, ensuring non-negative predictions. This is a Generalized Linear Model (GLM) with the Poisson as the response distribution.
>
> **2. Poisson Loss for Count Prediction:**
> Some gradient boosting implementations (XGBoost, LightGBM) offer a Poisson loss objective specifically for count data. This is the negative log-likelihood of the Poisson distribution.
>
> **3. Rare Event Modeling:**
> Fraud detection, equipment failures, and medical adverse events are often modeled with Poisson processes because the events are rare and roughly independent.

---

### Geometric Distribution

Models the number of trials until the **first** success.

**Why learn this**: "How many ads does a user see before clicking?" "How many cold emails until a response?" These are Geometric. The memoryless property also shows up in system reliability interviews — understanding it separates you from candidates who only memorize formulas.

**Used directly in**:
- **Expected trials to first event**: "How many calls until a sale?" → $E[X] = 1/p$. If close rate is 5%, expect 20 calls. Used in sales funnel modeling and capacity planning
- **Coupon collector problem**: "How many users must we sample to see all K segments?" is a sum of Geometric random variables — directly used in coverage analysis for data collection

$$X \sim \text{Geometric}(p)$$

$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

**Symbol definitions**:
- $X$ = the random variable: the total number of trials needed to get the first success
- $p$ = **parameter** you set: the probability of success on each trial
- $k$ = **query value** you plug in: "what if it takes exactly this many trials?"
- $(1-p)^{k-1}$ = probability of failing the first $k-1$ trials in a row
- $p$ (at the end) = probability of succeeding on the $k$-th trial

**What it outputs**: You give it a trial number $k$, and it returns the probability that the **first success happens on exactly trial $k$**. The formula literally reads: "fail $k-1$ times, then succeed once."

| Property | Value |
|----------|-------|
| **Parameters** | $p \in (0, 1]$ (success probability) |
| **Support** | $\{1, 2, 3, \ldots\}$ |
| **Mean** | $E[X] = 1/p$ |
| **Variance** | $\text{Var}(X) = (1-p)/p^2$ |

> [!WARNING]
> **Convention alert**: Some textbooks define Geometric as the number of *failures* before the first success (support starts at 0). Always check which convention is being used. The formulas above use the "number of trials" convention (support starts at 1).

#### The Memoryless Property

$$P(X > s + t \mid X > s) = P(X > t)$$

**Intuition**: If you have already failed $s$ times, your expected remaining wait is the same as if you just started. The past gives you no information about the future.

> [!TIP]
> **The Geometric distribution is the ONLY discrete distribution with the memoryless property.** Its continuous counterpart is the Exponential distribution (Section 2.2.2), which is the only continuous memoryless distribution.

> [!WARNING]
> **Common confusion: Independence $\neq$ Memorylessness**
>
> All Bernoulli-family distributions (Bernoulli, Binomial, Geometric, Negative Binomial) assume **independent (i.i.d.) trials** — each trial's outcome doesn't depend on previous trials. But **memorylessness** is a different, stronger property about the *aggregate random variable*, not individual trials.
>
> | Property | What it means | Who has it |
> |---|---|---|
> | **Independence (i.i.d.)** | Each trial's outcome doesn't depend on other trials | All Bernoulli-family distributions |
> | **Memorylessness** | The *waiting time* distribution is unchanged given past failures | Only **Geometric** (discrete) and **Exponential** (continuous) |
>
> **Why Binomial is NOT memoryless**: $X$ = total successes out of $n$ fixed trials. After observing some trials, you know how many remain and how many successes you've seen — that information changes your probability. There's nothing to "reset."
>
> **Why Negative Binomial is NOT memoryless**: $X$ = trials until $r$ successes. If you've already gotten 2 out of 3 needed successes, you only need 1 more — you're clearly closer to done. The past matters.
>
> **Why Geometric IS memoryless**: $X$ = trials until the **first** success. If you've failed 100 times, the probability of needing at least 5 more is *exactly the same* as when you started. Each trial is a fresh coin flip, and since you're waiting for only one success, there's no partial progress to track — you're always "starting from scratch."

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Expected Exploration Rounds (Agenda 9.2):**
> In reinforcement learning, if an agent has probability $p$ of succeeding at a task per attempt, the expected number of attempts until first success follows a Geometric distribution: $E[\text{attempts}] = 1/p$. This relates to the exploration-exploitation tradeoff.
>
> **2. Coupon Collector / Coverage Problems:**
> "How many samples do I need to see every class at least once?" is a sum of Geometric random variables — directly relevant to class-balanced sampling strategies.

---

### Negative Binomial Distribution

Models the number of trials until the $r$-th success. Generalizes the Geometric ($r = 1$).

**Why learn this**: Real count data (retail demand, hospital visits, insurance claims) is almost always overdispersed — variance exceeds the mean. Poisson can't handle this; Negative Binomial can. Amazon's DeepAR forecaster uses NB as its output distribution for exactly this reason.

**Used directly in**:
- **Overdispersed count regression**: `sm.GLM(y, X, family=sm.families.NegativeBinomial())` — use when Poisson fits poorly (variance >> mean), which is the norm in practice
- **Demand forecasting output**: Amazon's DeepAR, Facebook's Prophet (with uncertainty), and many probabilistic forecasters use NB as their output distribution for intermittent/bursty demand
- **Overdispersion test**: compare Poisson vs NB fit with a likelihood ratio test — if NB wins, your data is overdispersed and Poisson SEs are too narrow

$$X \sim \text{NegBin}(r, p)$$

$$P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, r+2, \ldots$$

**Symbol definitions**:
- $X$ = the random variable: the total number of trials needed to accumulate $r$ successes
- $r$ = **parameter** you set: how many successes you're waiting for
- $p$ = **parameter** you set: the probability of success on each trial
- $k$ = **query value** you plug in: "what if it takes exactly this many trials to get $r$ successes?"
- $\binom{k-1}{r-1}$ = the number of ways to arrange $r-1$ successes in the first $k-1$ trials (the $r$-th success is fixed at position $k$)
- $p^r$ = probability of $r$ successes, $(1-p)^{k-r}$ = probability of $k-r$ failures

**What it outputs**: You give it a trial count $k$, and it returns the probability that the $r$-th success occurs on exactly trial $k$. When $r=1$, this reduces to the Geometric distribution.

| Property | Value |
|----------|-------|
| **Parameters** | $r > 0$ (successes needed), $p \in (0, 1]$ (success probability) |
| **Support** | $\{r, r+1, r+2, \ldots\}$ |
| **Mean** | $E[X] = r/p$ |
| **Variance** | $\text{Var}(X) = r(1-p)/p^2$ |

> [!WARNING]
> **Scipy convention mismatch**: This guide defines $X$ = number of **trials** until $r$ successes (support starts at $r$). Scipy's `stats.nbinom(n, p)` defines $X$ = number of **failures** before $r$ successes (support starts at 0). To convert: `stats.nbinom.pmf(k - r, r, p)` gives the probability of needing $k$ total trials. The visualization code in this file uses this offset — don't be confused when `x_nb - r` appears.

> [!NOTE]
> **When Poisson fails, use Negative Binomial.** Recall that Poisson constrains mean = variance. In practice, count data is almost always **overdispersed** (variance > mean). The Negative Binomial adds an extra parameter to decouple mean and variance, making it far more flexible for real-world count data.
>
> | Distribution | Variance vs Mean | Use When |
> |---|---|---|
> | Poisson | $\text{Var} = \mu$ | Counts are well-behaved (rare) |
> | Negative Binomial | $\text{Var} > \mu$ | Counts are overdispersed (common) |

<details>
<summary><strong>Practical Guide: How to Tell from Your Data Whether to Use Poisson or Negative Binomial</strong></summary>

#### Step 1: Check the Dispersion Ratio

The single most important diagnostic. Compute variance / mean on your count data:

```python
import numpy as np

counts = np.array([...])  # your count data
dispersion_ratio = counts.var() / counts.mean()
print(f"Dispersion ratio: {dispersion_ratio:.2f}")
```

| Dispersion Ratio | Interpretation | Use |
|---|---|---|
| $\approx 1.0$ | Equidispersed — Poisson assumption holds | **Poisson** |
| $> 1.5$ | Overdispersed — variance exceeds mean | **Negative Binomial** |
| $< 0.8$ | Underdispersed — rare; consider Binomial or COM-Poisson | Neither |

**Plain English**: If your data has a mean of 5 events/day but the variance is 20, the dispersion ratio is 4.0 — far too high for Poisson. This happens when your data has more extreme values (or more zeros) than Poisson expects.

#### Step 2: Fit Both and Compare (Likelihood Ratio Test)

```python
import statsmodels.api as sm

# Fit Poisson
poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Fit Negative Binomial
nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

# Compare: lower AIC = better fit
print(f"Poisson AIC:  {poisson_model.aic:.1f}")
print(f"NegBin AIC:   {nb_model.aic:.1f}")

# If NegBin AIC is meaningfully lower, overdispersion is real
```

#### Step 3: Check Residuals

After fitting a Poisson model, look at Pearson residuals. If Poisson is correct, the residual deviance should be approximately equal to the residual degrees of freedom:

```python
# Pearson chi-squared / df should be near 1.0 for Poisson
pearson_chi2 = poisson_model.pearson_chi2
df_resid = poisson_model.df_resid
print(f"Pearson chi2 / df = {pearson_chi2 / df_resid:.2f}")
# If >> 1.0, your data is overdispersed -> switch to NB
```

#### Why This Matters Practically

Using Poisson on overdispersed data **underestimates standard errors**, which means:
- P-values are too small (false positives)
- Confidence intervals are too narrow (overconfident predictions)
- You'll conclude effects are "significant" when they're not

Switching to Negative Binomial fixes this by modeling the extra variance explicitly.

#### Real-World Examples

| Domain | Typical Data | Dispersion Ratio | Result |
|---|---|---|---|
| Server errors/hour | Steady, independent failures | $\approx 1.0$ | Poisson works |
| Daily retail demand | Bursty, driven by promotions/weather | $3-10$ | NB needed |
| Insurance claims/month | Many zeros, occasional large spikes | $5-50$ | NB needed |
| Hospital readmissions | Patient heterogeneity | $2-5$ | NB needed |

</details>

> [!NOTE]
> **Fundamental ML Connection**
>
> **Overdispersed Count Models (Agenda 2.3.4):**
> In practice (retail demand, click counts, hospital admissions), the Negative Binomial is often preferred over Poisson for regression because real count data almost always exhibits overdispersion. Many forecasting libraries (e.g., Amazon's DeepAR) use the Negative Binomial as their output distribution for exactly this reason.

---

### Multinomial Distribution

![Multinomial Distribution](./multinomial_distribution.png)

**Why learn this**: Every multi-class classifier (softmax output) is a Categorical distribution — Multinomial with $n=1$. The categorical cross-entropy loss you minimize in PyTorch is the negative log of this PMF. If you've trained a classifier, you've used this.

**Used directly in**:
- **Softmax cross-entropy loss**: `torch.nn.CrossEntropyLoss()` is the negative log of the Categorical PMF — every multi-class classifier training loop computes this
- **Topic modeling (LDA)**: each document's word distribution is Multinomial; `sklearn.decomposition.LatentDirichletAllocation` fits this model
- **A/B/n testing**: when comparing K > 2 variants, the allocation across variants follows a Multinomial — used in multi-arm experiment design

The multi-category extension of the Binomial. Models outcomes of $n$ trials, each falling into one of $K$ categories.

$$\mathbf{X} \sim \text{Multinomial}(n, \mathbf{p}) \quad \text{where } \mathbf{p} = (p_1, p_2, \ldots, p_K), \quad \sum_{k=1}^K p_k = 1$$

$$P(X_1 = x_1, \ldots, X_K = x_K) = \frac{n!}{x_1! x_2! \cdots x_K!} \prod_{k=1}^K p_k^{x_k}$$

**Symbol definitions**:
- $\mathbf{X} = (X_1, X_2, \ldots, X_K)$ = vector of random variables: the count of outcomes falling into each of $K$ categories
- $n$ = **parameter** you set: total number of trials
- $\mathbf{p} = (p_1, \ldots, p_K)$ = **parameter** you set: probability vector, where $p_k$ is the chance of landing in category $k$
- $x_1, \ldots, x_K$ = **query values** you plug in: "what if exactly $x_1$ land in category 1, $x_2$ in category 2, etc.?"
- $\frac{n!}{x_1! \cdots x_K!}$ = the multinomial coefficient: how many ways to arrange the outcomes into those category counts

**What it outputs**: You give it a vector of counts $(x_1, \ldots, x_K)$ that sum to $n$, and it returns the probability of exactly that allocation across all $K$ categories. Example: "What's the probability that out of 100 users, exactly 60 use iOS, 30 Android, 10 Web?"

| Property | Value |
|----------|-------|
| **Parameters** | $n$ (trials), $\mathbf{p}$ (probability vector, length $K$) |
| **Constraint** | $\sum_k x_k = n$ and $\sum_k p_k = 1$ |
| **Marginals** | Each $X_k \sim \text{Binomial}(n, p_k)$ |
| **Mean** | $E[X_k] = np_k$ |
| **Variance** | $\text{Var}(X_k) = np_k(1-p_k)$ |
| **Covariance** | $\text{Cov}(X_i, X_j) = -np_ip_j$ (negative — more of one means less of others) |

**Special cases**:
- $K = 2$: Multinomial reduces to **Binomial**
- $n = 1$: Single trial gives a **Categorical** distribution (Goodfellow calls this "Multinoulli")

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Multi-class Classification / Softmax (Agenda 3.1):**
> A softmax layer outputs a probability vector $\mathbf{p} = (p_1, \ldots, p_K)$ where $\sum p_k = 1$. This parameterizes a Categorical distribution (Multinomial with $n=1$): $P(Y=k \mid X) = p_k$. The categorical cross-entropy loss is the negative log-likelihood of this distribution:
> $$\mathcal{L} = -\sum_{k=1}^K y_k \log(p_k)$$
>
> **2. Topic Models / LDA (Agenda 3.2.3):**
> In Latent Dirichlet Allocation, each document's word distribution is Multinomial, with the probability vector drawn from a Dirichlet prior (the multivariate generalization of the Beta distribution).

---

### How Distributions Get Used in Practice

A distribution is a **machine with two modes**:

| Mode | You provide | It returns | Used for |
|---|---|---|---|
| **Evaluate** (PMF/PDF) | A specific value $k$ | The probability of that value | Loss functions, likelihoods, hypothesis testing |
| **Sample** (generate) | Just the parameters | A random value drawn from the distribution | Simulation, prediction, dropout masks, Thompson Sampling |

**Evaluate**: "What's the probability of exactly 3 server errors this hour?" → `stats.poisson.pmf(k=3, mu=5)` → returns $0.14$

**Sample**: "Simulate what happens this hour" → `stats.poisson.rvs(mu=5)` → returns a random count like $7$

When you **train** a model, you evaluate (compute likelihood of observed data). When you **simulate or predict**, you sample.

---

### Which Discrete Distribution? (Decision Guide)

```mermaid
flowchart TD
    START["You have<br/>COUNT DATA"]

    START --> Q1{"How many possible<br/>outcomes per trial?"}

    Q1 -->|"Exactly 2<br/>(yes/no, pass/fail,<br/>click/no-click)"| Q2{"How many<br/>trials?"}
    Q1 -->|"More than 2<br/>(K categories)"| MULTI["<strong>Multinomial</strong>(n, p)<br/>─────────────<br/>Mean: np_k<br/>Var: np_k(1-p_k)<br/>─────────────<br/>Example: device mix<br/>(iOS/Android/Web)<br/>ML: softmax output"]

    Q2 -->|"Exactly 1"| BERN["<strong>Bernoulli</strong>(p)<br/>─────────────<br/>Mean: p<br/>Var: p(1-p)<br/>─────────────<br/>Example: single user<br/>converts or not<br/>ML: sigmoid output,<br/>binary cross-entropy"]

    Q2 -->|"Fixed n trials,<br/>count total successes"| BINOM["<strong>Binomial</strong>(n, p)<br/>─────────────<br/>Mean: np<br/>Var: np(1-p)<br/>─────────────<br/>Example: 60 conversions<br/>out of 1000 visitors<br/>ML: A/B test,<br/>exact binomial test"]

    Q2 -->|"Unknown number of trials,<br/>waiting for 1st success"| GEOM["<strong>Geometric</strong>(p)<br/>─────────────<br/>Mean: 1/p<br/>Var: (1-p)/p^2<br/>─────────────<br/>Example: calls<br/>until first sale<br/>ONLY discrete<br/>memoryless dist."]

    Q2 -->|"Unknown number of trials,<br/>waiting for r-th success"| NEGBIN_WAIT["<strong>Neg. Binomial</strong>(r, p)<br/>─────────────<br/>Mean: r/p<br/>Var: r(1-p)/p^2<br/>─────────────<br/>Example: inspections<br/>until 3rd defect found<br/>Geometric when r=1"]

    START --> Q3{"Counting events in<br/>a fixed interval?<br/>(no fixed n)"}

    Q3 -->|"Yes"| Q4{"Check:<br/>Var / Mean ≈ ?"}
    Q4 -->|"≈ 1.0<br/>(equidispersed)"| POIS["<strong>Poisson</strong>(lambda)<br/>─────────────<br/>Mean: lambda<br/>Var: lambda<br/>─────────────<br/>Example: 3 server<br/>errors per hour<br/>ML: Poisson regression,<br/>count prediction loss"]
    Q4 -->|"> 1.5<br/>(overdispersed)"| NEGBIN_OD["<strong>Neg. Binomial</strong><br/>(overdispersed counts)<br/>─────────────<br/>Var > Mean<br/>─────────────<br/>Example: daily retail<br/>demand, insurance claims<br/>ML: DeepAR forecaster,<br/>NB regression"]

    style BERN fill:#4a90d9,stroke:#333,color:#fff
    style BINOM fill:#4a90d9,stroke:#333,color:#fff
    style POIS fill:#50c878,stroke:#333,color:#fff
    style GEOM fill:#ffa500,stroke:#333,color:#fff
    style NEGBIN_WAIT fill:#ffa500,stroke:#333,color:#fff
    style NEGBIN_OD fill:#50c878,stroke:#333,color:#fff
    style MULTI fill:#9370db,stroke:#333,color:#fff
    style Q4 fill:#f0f0f0,stroke:#666,color:#333
```

#### Quick-Reference Review: One-Liner Per Distribution

Use this table to self-test. Cover the right columns and try to recall from the name alone.

| Distribution | What is $X$? | Key Formula to Remember | When You'd Use It | Interview Trigger Phrase |
|---|---|---|---|---|
| **Bernoulli** | Single trial outcome (0 or 1) | $\text{Var} = p(1-p)$, max at $p=0.5$ | Model a single yes/no event | "binary cross-entropy loss" |
| **Binomial** | Count of successes in $n$ trials | $\text{Mean} = np$, $\text{SE} = \sqrt{p(1-p)/n}$ | A/B test: "how many converted?" | "sample size calculation" |
| **Poisson** | Count of events in fixed interval | $\text{Mean} = \text{Var} = \lambda$ | Server errors/hr, daily orders | "count data" or "rate" |
| **Geometric** | Trials until 1st success | $E[X] = 1/p$, memoryless | "How many calls until a sale?" | "waiting time" (discrete) |
| **Neg. Binomial** | Trials until $r$-th success, or overdispersed counts | $\text{Var} > \text{Mean}$ | Real count data (demand, claims) | "overdispersed" or "Var >> Mean" |
| **Multinomial** | Counts across $K$ categories | $\text{Cov}(X_i, X_j) = -np_ip_j$ | Multi-class allocation | "softmax" or "multi-class" |


---

### Discrete Distributions — Summary Table

| Distribution | PMF | Mean | Variance | Use Case |
|---|---|---|---|---|
| **Bernoulli**$(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | Single yes/no trial |
| **Binomial**$(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Count successes in $n$ trials |
| **Poisson**$(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Event counts in interval |
| **Geometric**$(p)$ | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | Trials until first success |
| **Neg. Binomial**$(r,p)$ | $\binom{k-1}{r-1}p^r(1-p)^{k-r}$ | $r/p$ | $r(1-p)/p^2$ | Trials until $r$-th success |
| **Multinomial**$(n,\mathbf{p})$ | $\frac{n!}{\prod x_k!}\prod p_k^{x_k}$ | $np_k$ | $np_k(1-p_k)$ | $n$ trials, $K$ categories |

---

### Python: Visualizing Discrete Distributions

![Discrete Distributions Gallery](./discrete_distributions_gallery.png)

<details>
<summary>Python Code for Visualization</summary>

```python
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
plt.show()
```

</details>

---

### Real-World Phenomena: Discrete Distributions

To bridge theory and practice, the following visualizes simulated real-world datasets alongside the theoretical distributions that best model them.

![Discrete Phenomena](./discrete_phenomena.png)

<details>
<summary>Python Code for Visualization</summary>

```python
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
plt.show()
```

</details>

#### Why This Distribution and Not Others? (Discrete)

| Phenomenon | Appropriate Distribution | Why this one? | Why not others? |
|---|---|---|---|
| **Daily Ad Conversions** | **Binomial** | We have a known, fixed number of daily visitors ($n$) and each operates independently with a fixed conversion rate ($p$). We want the total count. | Not **Poisson**, because Poisson assumes no upper bound on counts, while we know the max conversions cannot exceed daily visitors. Not **Geometric**, because we care about total count, not waiting time. |
| **Server Errors per Hour** | **Poisson** | Events (errors) happen independently over a continuous interval, and we don't know the exact number of total "trials" (web requests), just a consistent average rate ($\lambda$). | Not **Binomial**, because $n$ (total server requests) is massive and unbounded, while $p$ (chance of error per request) is tiny. Poisson is the natural limit here. Not **Normal**, because counts cannot be negative and are discrete. |
| **Sales Calls Until a Deal** | **Geometric** | We repeatedly dial independent leads until we get the *first* "Yes". We want to know how long we'll be waiting/trying. | Not **Binomial**, because $n$ (total trials) is not fixed in advance; the number of trials is the random variable itself. |
| **Ads Shown Until 5 Clicks** | **Negative Binomial** | We need a specific number of *multiple* successes ($r=5$). It is the sum of $r$ independent Geometric wait times. | Not **Poisson**, because we are measuring *trials elapsed* rather than *counts within a fixed time*. Not **Geometric**, because Geometric strictly models waiting for only *one* success. |
| **User Device Mix (iOS/Android/Web)** | **Multinomial** | A single pool of $n$ users falls into exactly one of $K > 2$ categories. | Not **Binomial**, because Binomial only handles binary buckets (e.g., just iOS vs Non-iOS). Multinomial naturally scales to $K$ buckets. |


---

#### Interview Priority: Discrete Distributions

| What to Know | Priority | Why |
|---|---|---|
| Bernoulli PMF, mean, variance | **Must know** | Foundation of binary classification, cross-entropy loss |
| Binomial: sum of Bernoullis, Normal approximation | **Must know** | A/B testing, hypothesis testing |
| Poisson: mean = variance, when to use | **Must know** | Count data modeling, common interview question |
| Poisson as Binomial limit | **Should know** | Shows mathematical depth |
| Geometric: memoryless property | **Should know** | Conceptual understanding, connects to Exponential |
| Negative Binomial: overdispersion fix for Poisson | **Should know** | Real-world modeling insight |
| Multinomial: softmax connection | **Must know** | Multi-class classification foundation |
---

## 2.2.2 Continuous Distributions **[H]**

> **Book**: Goodfellow Ch. 3.9.3-3.9.5 (pp. 63-65) | Gaussian, Exponential, Laplace, Dirac

> Continuous distributions describe measurements, durations, and neural network outputs. Most ML loss functions and generative models are built on these.

---

### Normal (Gaussian) Distribution

The most important distribution in all of statistics and ML.

**Why learn this**: MSE loss assumes Normal errors. Feature normalization assumes Normality. Confidence intervals, z-tests, and t-tests all rely on Normal approximations via CLT. Weight initialization in neural nets draws from Normals. This is genuinely the one distribution you cannot function without.

**Used directly in**:
- **MSE loss = Normal MLE**: minimizing MSE is equivalent to maximizing Normal log-likelihood — so every regression with MSE loss implicitly assumes Gaussian errors
- **Gaussian Naive Bayes**: `sklearn.naive_bayes.GaussianNB()` — assumes features are Normally distributed per class; surprisingly effective baseline for continuous features
- **Normality testing before parametric tests**: `scipy.stats.shapiro(data)` — run before t-tests on small samples; if it fails, switch to non-parametric alternatives
- **Q-Q plots for model diagnostics**: `sm.qqplot(residuals, line='s')` — checking if regression residuals are Normal is a core diagnostic step
- **Feature normalization**: `StandardScaler()` in sklearn transforms features to $\mathcal{N}(0,1)$ — required for SVMs, KNN, PCA, and neural networks

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Symbol definitions**:
- $X$ = the random variable: a continuous measurement (e.g., height, test score, residual error)
- $\mu$ = **parameter** you set: the mean (center) of the bell curve
- $\sigma^2$ = **parameter** you set: the variance (width) of the bell curve; $\sigma$ is the standard deviation
- $x$ = **query value** you plug in: any real number you want the density for
- $f(x)$ = the probability **density** at $x$ (not a probability itself — you must integrate over an interval to get a probability)

**What it outputs**: You give it a value $x$, and it returns the density — how relatively likely values near $x$ are. Higher density means observations are more concentrated around that value. The peak is always at $x = \mu$.

| Property | Value |
|----------|-------|
| **Parameters** | $\mu \in \mathbb{R}$ (mean/location), $\sigma^2 > 0$ (variance/spread) |
| **Support** | $(-\infty, +\infty)$ |
| **Mean** | $E[X] = \mu$ |
| **Variance** | $\text{Var}(X) = \sigma^2$ |
| **Mode = Mean = Median** | Symmetric: all three are $\mu$ |

#### The 68-95-99.7 Rule

| Range | Probability | Practical Meaning |
|-------|-------------|-------------------|
| $\mu \pm 1\sigma$ | 68.27% | ~2/3 of data |
| $\mu \pm 2\sigma$ | 95.45% | ~19/20 of data |
| $\mu \pm 3\sigma$ | 99.73% | Almost all data |

#### Standard Normal and Z-scores

The **standard Normal** is $Z \sim \mathcal{N}(0, 1)$. Any Normal can be standardized:

$$Z = \frac{X - \mu}{\sigma}$$

**Z-scores** measure "how many standard deviations from the mean." This is the foundation of hypothesis testing: convert a test statistic to a Z-score, then look up the probability.

#### Sum of Normals is Normal

If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are **independent**, then:

$$X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

> [!IMPORTANT]
> **Why the Normal is everywhere** (Goodfellow 3.9.3):
> 1. **CLT**: Sums of many independent random variables converge to Normal — regardless of the original distribution. This is why sample means, test statistics, and SGD gradients are approximately Normal.
> 2. **Maximum entropy (The Information Theory angle)**: Entropy measures unpredictability. If you only know the mean $\mu$ and variance $\sigma^2$ of some data, there are infinitely many possible distributions that fit those two facts. Which one should you pick?
>    - The Normal distribution mathematically has the **absolute highest entropy** (maximum uncertainty) among all distributions with a given mean and variance.
>    - **Why this matters for ML**: Choosing the maximum entropy distribution is the most conservative, "safest" choice you can make. It means you are assuming *only* the mean and variance, and absolutely nothing else (no hidden skew, no weird bounds). When we use MSE (which assumes Normal errors), we are formally stating: "I know my errors have some variance, but I refuse to inject any other assumptions or bias into my model." It is the ultimate "agnostic" distribution.
> 3. **Mathematical convenience**: The log of the Normal PDF is a quadratic — which makes MLE, MAP, and optimization easy.

<details>
<summary><strong>Worked Example: Feature Normalization</strong></summary>

**Why we normalize features**: If feature $X$ has mean $\mu = 1000$ and $\sigma = 500$, gradient descent will oscillate because the loss landscape is elongated. Z-scoring transforms it to $Z \sim \mathcal{N}(0, 1)$, making the landscape spherical.

```python
import numpy as np

# Raw feature with large scale
X = np.random.normal(loc=1000, scale=500, size=10000)
print(f"Before: mean={X.mean():.1f}, std={X.std():.1f}")

# Z-score normalization
Z = (X - X.mean()) / X.std()
print(f"After:  mean={Z.mean():.4f}, std={Z.std():.4f}")
# After: mean ~ 0.0000, std ~ 1.0000
```

</details>

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Weight Initialization (Agenda 3.3.1):**
> Neural network weights are typically initialized from $\mathcal{N}(0, \sigma^2)$ where $\sigma$ depends on fan-in/fan-out (Xavier: $\sigma^2 = 2/(n_{in} + n_{out})$, He: $\sigma^2 = 2/n_{in}$). The Normal choice ensures weights start symmetrically around zero with controlled variance.
>
> **2. Gaussian Noise and VAEs (Agenda 5.2):**
> Variational Autoencoders (VAEs) assume the latent space follows $\mathcal{N}(0, I)$. The "reparameterization trick" $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0,1)$ enables backpropagation through stochastic layers.
>
> **3. MSE Loss = Normal Likelihood (Agenda 3.1):**
> Minimizing Mean Squared Error is equivalent to maximizing the log-likelihood of a Normal distribution: $\text{MSE} = -\log \mathcal{N}(y \mid f(x), \sigma^2) + \text{const}$. This means when you use MSE loss, you are implicitly assuming your errors are Normally distributed.

---

### Exponential Distribution

Models the **waiting time** between events in a Poisson process.

**Why learn this**: "What's the expected time until the next server crash?" "How long until a customer churns?" These are Exponential. It's the simplest survival model and the continuous counterpart to Poisson — understanding the duality is a common interview question at Amazon and Netflix.

**Used directly in**:
- **Survival analysis baseline**: the simplest survival model assumes constant hazard $h(t) = \lambda$, which gives Exponential survival times — it's the null model you compare more complex models (Weibull, Cox PH) against
- **Queueing theory / capacity planning**: inter-arrival times in an M/M/1 queue are Exponential — used to model customer wait times, server request spacing, and SLA analysis
- **Time-to-event feature engineering**: for user churn models, "time since last login" often follows an Exponential — recognizing this guides your feature transformation choices

$$X \sim \text{Exponential}(\lambda)$$

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**Symbol definitions**:
- $X$ = the random variable: the waiting time until the next event (e.g., minutes until next server error)
- $\lambda$ = **parameter** you set: the rate of events per unit time (e.g., 3 errors/hour)
- $x$ = **query value** you plug in: a specific waiting time (must be $\geq 0$)
- $e^{-\lambda x}$ = exponential decay — longer waits are increasingly unlikely

**What it outputs**: You give it a time $x$, and it returns the density at that waiting time. Short waits are most likely (density is highest at $x=0$ and decays). Note: the mean wait is $1/\lambda$, so a higher rate means shorter expected waits.

| Property | Value |
|----------|-------|
| **Parameters** | $\lambda > 0$ (rate) |
| **Support** | $[0, +\infty)$ |
| **Mean** | $E[X] = 1/\lambda$ |
| **Variance** | $\text{Var}(X) = 1/\lambda^2$ |
| **Memoryless** | $P(X > s + t \mid X > s) = P(X > t)$ |

> [!TIP]
> **Poisson-Exponential duality**: If events arrive at rate $\lambda$ per unit time (Poisson), then the time *between* consecutive events follows $\text{Exponential}(\lambda)$. They are two views of the same process:
>
> | Poisson | Exponential |
> |---------|-------------|
> | Counts events in fixed time | Measures time between events |
> | Discrete (counts) | Continuous (time) |
> | Parameter $\lambda$ = rate | Parameter $\lambda$ = same rate |
> | Mean = $\lambda$ events | Mean = $1/\lambda$ time |

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Survival Analysis (Agenda 9.3):**
> The Exponential distribution is the simplest survival model: the "hazard rate" is constant $h(t) = \lambda$. This is the baseline for more flexible models like Weibull or Cox proportional hazards. When they say "the event has no memory," this is the Exponential assumption.
>
> **2. Laplace Distribution and L1 Regularization (Section 2.1.5):**
> Goodfellow (3.9.4) discusses the Laplace distribution, which is a "double Exponential" (Exponential on both sides of 0). Using a Laplace prior in MAP estimation gives L1 (Lasso) regularization — producing sparse solutions.

---

### Uniform Distribution

![Uniform Distribution](./uniform_distribution.png)

Equal probability across an interval — the distribution of "no information."

**Why learn this**: Random hyperparameter search (learning rate, dropout rate) samples from Uniform. More importantly, Uniform prior = MLE — understanding this connects Bayesian and frequentist thinking, which is a hallmark of statistical maturity in interviews.

**Used directly in**:
- **Random hyperparameter search**: `scipy.stats.uniform(loc, scale)` as the distribution for `RandomizedSearchCV` — sampling learning rates, dropout rates, and regularization strengths
- **Log-uniform for scale parameters**: learning rate search is typically `loguniform(1e-5, 1e-1)` — uniform in log-space, which is `scipy.stats.loguniform`
- **Random initialization**: Xavier/He initialization draws from scaled Uniform — `torch.nn.init.kaiming_uniform_()`

$$X \sim \text{Uniform}(a, b)$$

$$f(x) = \frac{1}{b - a}, \quad a \leq x \leq b$$

**Symbol definitions**:
- $X$ = the random variable: a value equally likely to land anywhere in $[a, b]$
- $a, b$ = **parameters** you set: the left and right endpoints of the interval
- $x$ = **query value** you plug in: any value between $a$ and $b$
- $\frac{1}{b-a}$ = constant density — every point in the interval is equally likely

**What it outputs**: For any $x$ in $[a, b]$, the density is the same constant $\frac{1}{b-a}$. Outside $[a, b]$, the density is 0. This is the "I have no idea, every value is equally plausible" distribution.

| Property | Value |
|----------|-------|
| **Parameters** | $a, b \in \mathbb{R}$, $a < b$ (endpoints) |
| **Support** | $[a, b]$ |
| **Mean** | $E[X] = (a + b) / 2$ |
| **Variance** | $\text{Var}(X) = (b - a)^2 / 12$ |

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Random Initialization and Random Search:**
> Uniform distributions are used for random hyperparameter search (sampling learning rates uniformly in log-space), random seeds, and random projections.
>
> **2. Uniform Prior = MLE (Section 2.1.5):**
> A Uniform prior $P(\theta) = \text{const}$ contributes nothing to the MAP objective, so MAP with a Uniform prior reduces to MLE. This is the formal justification for "MLE assumes no prior knowledge."

---

### Log-Normal Distribution

If $\log(X) \sim \mathcal{N}(\mu, \sigma^2)$, then $X \sim \text{Log-Normal}(\mu, \sigma^2)$.

**Why learn this**: House prices, salaries, response latencies, and stock returns are all Log-Normal. When your regression target is right-skewed and positive, log-transforming it before fitting is standard practice — and understanding why (Jensen's inequality, back-transformation bias) separates good practitioners from great ones.

**Used directly in**:
- **Target transformation for regression**: `np.log1p(y)` before fitting, `np.expm1(pred)` after — standard practice for right-skewed targets (house prices, salaries, latencies). Remember: $E[e^Z] \neq e^{E[Z]}$ (apply smearing correction)
- **Latency / SLA modeling**: P99 latency analysis often assumes Log-Normal — `scipy.stats.lognorm.ppf(0.99, s, scale)` gives the 99th percentile directly
- **Financial modeling**: stock returns are approximately Log-Normal — if you model price paths (Monte Carlo simulation), this is the assumed distribution

$$f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

**Symbol definitions**:
- $X$ = the random variable: a strictly positive, right-skewed measurement (e.g., house price, response latency, salary)
- $\mu$ = **parameter**: the mean of $\ln(X)$ (not the mean of $X$ itself!)
- $\sigma^2$ = **parameter**: the variance of $\ln(X)$
- $x$ = **query value** you plug in: any positive real number
- $\ln x$ = natural log of $x$ — this transformation maps Log-Normal back to Normal

**What it outputs**: You give it a positive value $x$, and it returns the density. Key insight: if you take the log of a Log-Normal random variable, you get a Normal. So the distribution is just a Normal "seen through the exponential lens."

| Property | Value |
|----------|-------|
| **Parameters** | $\mu \in \mathbb{R}$, $\sigma^2 > 0$ (of the underlying Normal) |
| **Support** | $(0, +\infty)$ |
| **Mean** | $E[X] = e^{\mu + \sigma^2/2}$ |
| **Variance** | $\text{Var}(X) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$ |
| **Shape** | Right-skewed, always positive |

> [!NOTE]
> **When to suspect Log-Normal**: If your data is strictly positive and right-skewed — and especially if a *multiplicative* process generates it (e.g., stock returns, income, city sizes, time-to-failure) — try taking the log. If $\log(X)$ looks Normal, your data is Log-Normal.

> [!NOTE]
> **Fundamental ML Connection**
>
> **Log-transforming Targets in Regression:**
> When predicting right-skewed targets (house prices, salaries, response times), applying $\log(y)$ before fitting often improves model performance because the transformed target is closer to Normal — satisfying the implicit MSE/Normal assumption. Remember to exponentiate predictions back and that $E[e^Z] \neq e^{E[Z]}$ (Jensen's inequality from Section 2.1.6).

---

### Beta Distribution

The distribution over probabilities — values constrained to $[0, 1]$.

**Why learn this**: This is the engine of Bayesian A/B testing and Thompson Sampling. At any company running experiments, you'll either use or be asked about Beta-Binomial conjugacy. If you understand Beta, you can explain why Bayesian A/B testing doesn't require fixed sample sizes — a massive practical advantage.

**Used directly in**:
- **Thompson Sampling implementation**: for each arm, sample from `np.random.beta(alpha + successes, beta + failures)` and pick the highest — this is the complete algorithm for Bayesian A/B testing
- **Bayesian posterior updates**: `posterior = Beta(prior_alpha + k, prior_beta + n - k)` after observing k successes in n trials — no MCMC needed, closed-form
- **Calibration analysis**: model predicted probabilities should follow a Beta-like distribution between 0 and 1; `scipy.stats.beta.fit(predicted_probs)` helps diagnose calibration issues

$$X \sim \text{Beta}(\alpha, \beta)$$

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1$$

**Symbol definitions**:
- $X$ = the random variable: a probability or proportion (always between 0 and 1)
- $\alpha$ = **parameter** you set: controls mass toward 1 (think: "pseudo-count of successes")
- $\beta$ = **parameter** you set: controls mass toward 0 (think: "pseudo-count of failures")
- $x$ = **query value** you plug in: a specific probability value in $[0, 1]$
- $B(\alpha, \beta)$ = the Beta function — a normalizing constant that ensures the PDF integrates to 1

**What it outputs**: You give it a probability $x \in [0, 1]$, and it returns the density — how plausible that probability value is given your prior beliefs encoded in $\alpha$ and $\beta$. This is the distribution you use to express uncertainty *about a probability itself*.

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function (normalizing constant).

| Property | Value |
|----------|-------|
| **Parameters** | $\alpha > 0$, $\beta > 0$ (shape parameters) |
| **Support** | $[0, 1]$ |
| **Mean** | $E[X] = \frac{\alpha}{\alpha + \beta}$ |
| **Variance** | $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| **Mode** | $\frac{\alpha - 1}{\alpha + \beta - 2}$ (for $\alpha, \beta > 1$) |

#### Shape Gallery

The Beta distribution is incredibly flexible — it can be uniform, U-shaped, skewed, or symmetric:

| $\alpha$ | $\beta$ | Shape | Interpretation |
|----------|---------|-------|----------------|
| 1 | 1 | Uniform (flat) | No prior information |
| 0.5 | 0.5 | U-shaped (edges) | Extreme values likely |
| 2 | 5 | Right-skewed (mass near 0, tail toward 1) | Low values more likely |
| 5 | 2 | Left-skewed (mass near 1, tail toward 0) | High values more likely |
| 5 | 5 | Symmetric bell | Centered around 0.5 |
| 50 | 50 | Tight bell | Strong belief around 0.5 |

> [!TIP]
> **Mental model**: Think of $\alpha$ as "pseudo-counts of successes" and $\beta$ as "pseudo-counts of failures." $\text{Beta}(1, 1)$ = "I've seen nothing" (Uniform). $\text{Beta}(10, 2)$ = "I've seen 10 successes and 2 failures" (skewed right, probably $p \approx 0.83$).

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Bayesian A/B Testing / Thompson Sampling (Agenda 1.2.1):**
> The Beta distribution is the conjugate prior for the Bernoulli/Binomial likelihood. Start with $\text{Beta}(\alpha_0, \beta_0)$ as your prior belief about a conversion rate. After observing $s$ successes and $f$ failures, the posterior is $\text{Beta}(\alpha_0 + s, \beta_0 + f)$. Thompson Sampling draws a sample from each arm's posterior Beta and picks the highest — a simple yet optimal exploration strategy.
>
> **2. Beta-Binomial Conjugacy (Section 2.1.4):**
> This is the most important conjugacy pair in applied Bayesian inference. It enables closed-form posterior updates without MCMC, which is why Bayesian A/B testing is computationally cheap.

![Beta Distribution Gallery](./beta_distribution_gallery.png)

<details>
<summary>Python Code for Visualization</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Beta Distribution Shape Gallery', fontsize=16, fontweight='bold')

params = [
    (1, 1, 'Uniform: Beta(1,1)'),
    (0.5, 0.5, 'U-shaped: Beta(0.5,0.5)'),
    (2, 5, 'Right-skewed: Beta(2,5)'),
    (5, 2, 'Left-skewed: Beta(5,2)'),
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
plt.show()
```

</details>

---

### Gamma Distribution

![Gamma Distribution](./gamma_distribution.png)

A flexible family for positive-valued random variables. Generalizes the Exponential.

**Why learn this**: Gamma appears as the conjugate prior for Poisson rates (Bayesian demand modeling) and as the parent family that connects Exponential and Chi-squared. Knowing this relationship lets you navigate the testing distribution hierarchy fluently.

**Used directly in**:
- **Bayesian prior for rates**: in PyMC / Stan, `pm.Gamma('rate', alpha=2, beta=1)` is a standard weakly informative prior for Poisson rate parameters
- **Insurance / claims modeling**: Gamma regression `sm.GLM(y, X, family=sm.families.Gamma())` for positive-valued, right-skewed outcomes (claim amounts, repair costs)
- **Waiting time aggregation**: if you're modeling total time for $k$ sequential service steps (each Exponential), the sum is Gamma — used in operations research

$$X \sim \text{Gamma}(\alpha, \beta)$$

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

**Symbol definitions**:
- $X$ = the random variable: a positive-valued continuous quantity (e.g., total wait time, insurance claim amount)
- $\alpha$ = **parameter** you set: the shape (controls skewness; higher $\alpha$ makes it more symmetric)
- $\beta$ = **parameter** you set: the rate (higher $\beta$ compresses the distribution leftward)
- $x$ = **query value** you plug in: any positive real number
- $\Gamma(\alpha)$ = the Gamma function (a generalization of factorial: $\Gamma(n) = (n-1)!$ for integers)

**What it outputs**: You give it a positive value $x$, and it returns the density. The Gamma is a flexible family: it includes the Exponential ($\alpha=1$) and Chi-squared ($\alpha=k/2, \beta=1/2$) as special cases. Think of it as "time to complete $\alpha$ stages, each with rate $\beta$."

| Property | Value |
|----------|-------|
| **Parameters** | $\alpha > 0$ (shape), $\beta > 0$ (rate) |
| **Support** | $(0, +\infty)$ |
| **Mean** | $E[X] = \alpha/\beta$ |
| **Variance** | $\text{Var}(X) = \alpha/\beta^2$ |

**Key special cases**:
- $\text{Gamma}(1, \lambda)$ = $\text{Exponential}(\lambda)$
- $\text{Gamma}(n/2, 1/2)$ = $\chi^2(n)$ (Chi-squared with $n$ degrees of freedom)
- Sum of $n$ i.i.d. $\text{Exponential}(\beta)$ = $\text{Gamma}(n, \beta)$

> [!WARNING]
> **Scipy convention**: This guide uses the **rate** parameterization $\text{Gamma}(\alpha, \beta)$ where $\beta$ is the rate. Scipy uses the **shape/scale** parameterization: `stats.gamma(a=alpha, scale=1/beta)`. Always pass `scale=1/beta` — not `beta` directly. Forgetting this is a common source of bugs.

> [!NOTE]
> **Fundamental ML Connection**
>
> **Bayesian Prior for Rate Parameters:**
> The Gamma distribution is the conjugate prior for the Poisson rate $\lambda$ and the precision (inverse variance) of a Normal. When you see "Gamma prior" in Bayesian modeling, it's placing a belief on a positive-valued parameter.

---

### Chi-Squared Distribution

The distribution of the sum of squared standard Normals. Foundation of hypothesis testing.

**Why learn this**: Every time you run a chi-squared test of independence for feature selection, compute a goodness-of-fit test, or build a confidence interval for variance, you're using this. It's also the building block for the t and F distributions used in regression.

**Used directly in**:
- **Feature selection**: `sklearn.feature_selection.chi2(X, y)` scores categorical features by their chi-squared statistic against the target — a fast filter method before training
- **Goodness-of-fit testing**: `scipy.stats.chisquare(observed, expected)` — tests if observed category counts match a hypothesized distribution (e.g., "is traffic evenly split across buckets?")
- **Model calibration check**: Hosmer-Lemeshow test groups predictions into deciles and uses chi-squared to test if observed rates match predicted rates

$$X \sim \chi^2(k)$$

If $Z_1, Z_2, \ldots, Z_k \sim \mathcal{N}(0,1)$ are i.i.d., then:

$$X = \sum_{i=1}^k Z_i^2 \sim \chi^2(k)$$

**Symbol definitions**:
- $X$ = the random variable: the sum of $k$ squared standard Normal values
- $k$ = **parameter**: degrees of freedom (the number of independent standard Normals being squared and summed)
- $Z_i$ = independent standard Normal random variables $\mathcal{N}(0,1)$

**What it outputs**: The Chi-squared is not one you typically query directly for density. Instead it's used as a **reference distribution for test statistics**. You compute a chi-squared test statistic from data, then ask: "How likely is a value this large under $H_0$?" That p-value comes from the $\chi^2(k)$ distribution.

| Property | Value |
|----------|-------|
| **Parameters** | $k \in \mathbb{N}$ (degrees of freedom) |
| **Support** | $[0, +\infty)$ |
| **Mean** | $E[X] = k$ |
| **Variance** | $\text{Var}(X) = 2k$ |
| **Special case** | $\chi^2(k) = \text{Gamma}(k/2, 1/2)$ |

> [!NOTE]
> **Degrees of freedom intuition**: The parameter $k$ counts how many independent "pieces of information" contribute to the sum. When fitting a model with $p$ parameters from $n$ data points, the residual sum of squares (after dividing by $\sigma^2$) follows $\chi^2(n-p)$. That's why we divide by $n-1$ (not $n$) for sample variance — we "used up" 1 degree of freedom estimating the mean.

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Chi-squared Test of Independence (Agenda 2.3.3):**
> Tests whether two categorical features are independent. The test statistic $\sum \frac{(O - E)^2}{E}$ follows $\chi^2$ under $H_0$. Used for feature selection in categorical data.
>
> **2. Goodness-of-Fit Testing:**
> Tests whether observed counts match a hypothesized distribution. Same $\chi^2$ statistic, different application.
>
> **3. Connection to Sample Variance:**
> $(n-1)s^2/\sigma^2 \sim \chi^2(n-1)$ — this is why confidence intervals for variance use the Chi-squared distribution.

---

### t-Distribution

Arises when estimating a Normal mean from a small sample with unknown variance. Heavier tails than Normal.

**Why learn this**: Every regression p-value you've ever seen was computed from a t-distribution. Every small-sample A/B test uses t-tests. "When do you use t vs z?" is asked in nearly every AS interview. Heavy tails also make the t-distribution a basis for robust regression.

**Used directly in**:
- **Every regression p-value**: `model.pvalues` in statsmodels computes $t = \hat{\beta}/SE(\hat{\beta})$ and looks up the t-distribution — this is how you know if a coefficient is "significant"
- **Small-sample A/B tests**: `scipy.stats.ttest_ind(treatment, control)` — the default two-sample comparison for continuous metrics
- **Robust regression**: PyMC's `pm.StudentT` likelihood with low df gives a regression that automatically downweights outliers — used when Normal errors assumption is too fragile

$$X \sim t(k)$$

If $Z \sim \mathcal{N}(0,1)$ and $V \sim \chi^2(k)$ are independent, then:

$$T = \frac{Z}{\sqrt{V/k}} \sim t(k)$$

**Symbol definitions**:
- $T$ = the random variable: a ratio of a standard Normal to the square root of a Chi-squared (scaled)
- $Z$ = a standard Normal $\mathcal{N}(0,1)$ random variable
- $V$ = a $\chi^2(k)$ random variable, independent of $Z$
- $k$ = **parameter**: degrees of freedom (in practice, typically $n - 1$ or $n - p$ from your sample)

**What it outputs**: Like Chi-squared, this is a **reference distribution for test statistics**. In practice: you compute $t = \hat{\beta}/SE(\hat{\beta})$, then look up how extreme that value is under the $t(k)$ distribution. The result is your p-value. It looks like a Normal but with heavier tails (more probability in the extremes), which accounts for the extra uncertainty from estimating variance.

| Property | Value |
|----------|-------|
| **Parameters** | $k \in \mathbb{N}$ (degrees of freedom) |
| **Support** | $(-\infty, +\infty)$ |
| **Mean** | $E[X] = 0$ (for $k > 1$) |
| **Variance** | $\text{Var}(X) = k/(k-2)$ (for $k > 2$) |
| **Key behavior** | Heavier tails than Normal; approaches $\mathcal{N}(0,1)$ as $k \to \infty$ |

> [!WARNING]
> **When to use t vs Normal**:
>
> | Scenario | Use | Why |
> |----------|-----|-----|
> | Large sample ($n > 30$), known $\sigma$ | Z-test (Normal) | CLT + known variance |
> | Small sample, unknown $\sigma$ | t-test (t-distribution) | Extra uncertainty from estimating $\sigma$ adds heavier tails |
> | Any sample, unknown $\sigma$ | t-test (safe choice) | Always valid; reduces to Z-test for large $n$ |

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Regression Coefficient Testing (Agenda 2.3.4):**
> In OLS regression, each coefficient's test statistic $t = \hat{\beta}/\text{SE}(\hat{\beta})$ follows a t-distribution under $H_0: \beta = 0$. This is how p-values for regression coefficients are computed.
>
> **2. Robust Loss Functions:**
> The t-distribution's heavy tails make it more robust to outliers than the Normal. Some robust regression methods assume t-distributed errors instead of Normal errors, downweighting outliers automatically.

---

### F-Distribution

The ratio of two independent Chi-squared variables (divided by their degrees of freedom).

**Why learn this**: "Does this model explain significantly more variance than a baseline?" — that's an F-test. ANOVA uses F. The overall regression significance test uses F. When comparing nested models (did adding these features help?), F is the answer.

**Used directly in**:
- **Overall model significance**: `model.f_pvalue` in statsmodels — tests whether your regression explains more variance than a mean-only model (should always check this)
- **Nested model comparison**: "Did adding these 3 features significantly improve the model?" — F-test via `sm.stats.anova_lm(model_reduced, model_full)`
- **ANOVA for A/B/n tests**: `scipy.stats.f_oneway(group_a, group_b, group_c)` — testing if any variant has a different mean

$$X \sim F(d_1, d_2)$$

If $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent, then:

$$F = \frac{U/d_1}{V/d_2} \sim F(d_1, d_2)$$

**Symbol definitions**:
- $F$ = the random variable: a ratio of two scaled Chi-squared variables
- $U$ = a $\chi^2(d_1)$ random variable (numerator)
- $V$ = a $\chi^2(d_2)$ random variable (denominator), independent of $U$
- $d_1$ = **parameter**: numerator degrees of freedom (typically the number of predictors being tested)
- $d_2$ = **parameter**: denominator degrees of freedom (typically $n - p$, sample size minus number of parameters)

**What it outputs**: Another **reference distribution for test statistics**. You compute an F-statistic (ratio of "explained variance" to "unexplained variance"), then look up the p-value from the $F(d_1, d_2)$ distribution. A large $F$ means the model explains significantly more than noise. Used in ANOVA and overall regression significance tests.

| Property | Value |
|----------|-------|
| **Parameters** | $d_1, d_2$ (numerator and denominator degrees of freedom) |
| **Support** | $[0, +\infty)$ |
| **Mean** | $E[X] = d_2/(d_2-2)$ (for $d_2 > 2$) |
| **Key relationship** | If $T \sim t(k)$, then $T^2 \sim F(1, k)$ |

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. ANOVA F-test (Agenda 2.3.3):**
> The F-statistic compares between-group variance to within-group variance. A large F means the groups are significantly different. $F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$.
>
> **2. Regression F-test (Agenda 2.3.4):**
> The overall F-test for regression asks: "Does this model explain significantly more variance than a mean-only model?" It tests whether *all* coefficients are jointly zero.

---

### Dirac Delta Distribution

![Dirac Delta Distribution](./dirac_delta.png)

<details>
<summary>Python Code for Visualization</summary>

```python
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
plt.show()
```

</details>

**Why learn this**: The Dirac delta formalizes the empirical distribution — the thing you actually compute when you take sample means. It's also how mixture models and kernel density estimation are built. Understanding it shows theoretical depth in interviews.

**Used directly in**:
- **Kernel Density Estimation (KDE)**: KDE smooths Dirac deltas with a kernel — `scipy.stats.gaussian_kde(data)` is literally replacing point masses with Gaussians to get a smooth density estimate
- **Empirical CDF**: `statsmodels.distributions.ECDF(data)` constructs the step function from Dirac deltas — used for comparing distributions and the Kolmogorov-Smirnov test

A "distribution" that puts all its mass at a single point. Not a distribution in the classical sense, but extremely useful.

$$p(x) = \delta(x - \mu)$$

where $\delta(x) = 0$ for $x \neq 0$ and $\int \delta(x)dx = 1$.

| Property | Value |
|----------|-------|
| **Parameters** | $\mu$ (the point mass location) |
| **Mean** | $E[X] = \mu$ |
| **Variance** | $\text{Var}(X) = 0$ |

> [!NOTE]
> **Fundamental ML Connection** (Goodfellow 3.9.5)
>
> **Empirical Distribution:**
> Given data points $\{x_1, \ldots, x_n\}$, the empirical distribution is a mixture of Dirac deltas:
> $$\hat{p}(x) = \frac{1}{n} \sum_{i=1}^n \delta(x - x_i)$$
> This connects to MLE: the empirical distribution is the *maximum likelihood* distribution — it assigns equal probability to each observed data point and zero to everything else. When we compute sample means, we are computing expectations under this empirical distribution.

---

### Multivariate Normal Distribution

The multi-dimensional generalization of the Normal. The most important multivariate distribution.

**Why learn this**: PCA finds the eigenvectors of the MVN covariance matrix. Gaussian Processes output MVN predictions. LDA assumes MVN class-conditionals. Mahalanobis distance (from MVN) is used for anomaly detection. If you work with multivariate data, you work with this.

**Used directly in**:
- **PCA**: `sklearn.decomposition.PCA` finds the eigenvectors of the sample covariance matrix — the principal components are the directions of maximum variance in the MVN
- **Mahalanobis anomaly detection**: `scipy.spatial.distance.mahalanobis(x, mean, cov_inv)` — flag data points that are far from the center accounting for correlations; used in fraud detection and quality control
- **Gaussian Processes**: `sklearn.gaussian_process.GaussianProcessRegressor` — predictions are MVN conditioned on training data, giving both point estimates and uncertainty bands
- **Linear Discriminant Analysis**: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` assumes MVN per class with shared covariance — an interpretable classifier that's optimal under these assumptions

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

| Property | Value |
|----------|-------|
| **Parameters** | $\boldsymbol{\mu} \in \mathbb{R}^d$ (mean vector), $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ (covariance matrix, PSD) |
| **Support** | $\mathbb{R}^d$ |
| **Marginals** | Each $X_i \sim \mathcal{N}(\mu_i, \Sigma_{ii})$ |
| **Conditional** | $X_1 \mid X_2 = x_2$ is also multivariate Normal |
| **Key quantity** | Precision matrix $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ |

#### Covariance Matrix Shapes

The covariance matrix $\boldsymbol{\Sigma}$ controls the shape and orientation of the distribution:

| $\boldsymbol{\Sigma}$ Shape | Distribution Shape | Example |
|---|---|---|
| $\sigma^2 \mathbf{I}$ (scaled identity) | Spherical (circular contours) | Independent features, equal variance |
| Diagonal (different entries) | Axis-aligned ellipse | Independent features, different variances |
| Full (off-diagonal entries) | Rotated ellipse | Correlated features |

#### Mahalanobis Distance

$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

**Intuition**: Euclidean distance "adjusted" for correlations. A point that is 3 units away in a high-variance direction is less unusual than 3 units away in a low-variance direction. Mahalanobis distance accounts for this.

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. PCA (Agenda 3.2.2):**
> PCA finds the eigenvectors of the covariance matrix $\boldsymbol{\Sigma}$. The top-$k$ eigenvectors (directions of maximum variance) define the principal components. The eigenvalues tell you how much variance each component explains.
>
> **2. Gaussian Processes (Agenda 3.5):**
> A GP is an infinite-dimensional multivariate Normal. The kernel function $k(x_i, x_j)$ defines the covariance matrix entries $\Sigma_{ij}$. Conditioning on observed data gives posterior predictions that are also multivariate Normal — with closed-form mean and variance.
>
> **3. Linear Discriminant Analysis (LDA):**
> LDA assumes each class has a multivariate Normal distribution with different means but shared covariance. The decision boundary is where the class-conditional densities are equal — which turns out to be a linear function when $\boldsymbol{\Sigma}$ is shared.
>
> **4. Anomaly Detection:**
> Points with large Mahalanobis distance from the data center are outliers. This is more sophisticated than simple distance thresholds because it accounts for the correlation structure of the data.

![Multivariate Normal Contours](./multivariate_normal_contours.png)

<details>
<summary>Python Code for Visualization</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Multivariate Normal: Effect of Covariance', fontsize=16, fontweight='bold')

# Create grid
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Three different covariance matrices
covs = [
    ([[1, 0], [0, 1]], 'Spherical\nSigma = I'),
    ([[2, 0], [0, 0.5]], 'Axis-aligned\nSigma = diag(2, 0.5)'),
    ([[2, 1.2], [1.2, 1]], 'Correlated\nSigma = [[2,1.2],[1.2,1]]')
]

for ax, (cov, title) in zip(axes, covs):
    rv = stats.multivariate_normal([0, 0], cov)
    ax.contour(X, Y, rv.pdf(pos), levels=8, cmap='Blues')
    ax.contourf(X, Y, rv.pdf(pos), levels=8, cmap='Blues', alpha=0.4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig('multivariate_normal_contours.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

---

### Which Continuous Distribution? (Decision Guide)

```mermaid
flowchart TD
    START["You have<br/>CONTINUOUS DATA"]
    
    START --> Q1{"What is the<br/>support (range)?"}
    
    Q1 -->|"Bounded [0,1]"| BETA["<strong>Beta</strong>(alpha, beta)<br/>─────────────<br/>Mean: alpha/(alpha+beta)<br/>─────────────<br/>Example: CTR, batting avg<br/>ML: Bayesian A/B testing,<br/>Thompson Sampling"]
    
    Q1 -->|"Bounded [a,b]"| UNIF["<strong>Uniform</strong>(a, b)<br/>─────────────<br/>Flat density: 1/(b-a)<br/>─────────────<br/>Example: spinning a dial,<br/>true random noise<br/>ML: Hyperparameter search"]
    
    Q1 -->|"All real numbers<br/>(-inf, +inf)"| Q2{"Symmetric<br/>or Heavy-tailed?"}
    Q1 -->|"Positive only<br/>(0, +inf)"| Q3{"What generated<br/>the data?"}
    
    Q2 -->|"Symmetric,<br/>known variance"| NORM["<strong>Normal</strong>(mu, sigma^2)<br/>─────────────<br/>Mean: mu  |  Var: sigma^2<br/>─────────────<br/>Example: human heights,<br/>measurement errors<br/>ML: MSE loss = Normal MLE"]
    
    Q2 -->|"Symmetric,<br/>unknown variance<br/>(small sample)"| TDIST["<strong>t-distribution</strong>(k)<br/>─────────────<br/>Heavier tails than Normal<br/>─────────────<br/>Example: stock returns,<br/>small-sample estimates<br/>ML: Regression p-values"]
    
    Q2 -->|"Need multi-<br/>dimensional"| MVN["<strong>Multivariate N.</strong>(mu, Sigma)<br/>─────────────<br/>Uses Covariance matrix<br/>─────────────<br/>Example: GPS coordinates,<br/>height & weight together<br/>ML: PCA, Gaussian Processes"]
    
    Q3 -->|"Waiting time<br/>between events"| EXP["<strong>Exponential</strong>(lambda)<br/>─────────────<br/>Mean: 1/lambda<br/>─────────────<br/>Example: server uptime,<br/>radioactive decay<br/>ONLY continuous memoryless dist."]
    
    Q3 -->|"Multiplicative<br/>growth/decay"| LOGN["<strong>Log-Normal</strong>(mu, sigma)<br/>─────────────<br/>Right-skewed<br/>─────────────<br/>Example: salaries, house<br/>prices, internet traffic<br/>ML: Target log-transform"]
    
    Q3 -->|"Sum of squared<br/>Normals"| CHI["<strong>Chi-squared</strong>(k)<br/>─────────────<br/>Mean: k (degrees of freedom)<br/>─────────────<br/>Example: sum of squared errors,<br/>sample variance dispersion<br/>ML: Goodness-of-fit tests"]
    
    Q3 -->|"Ratio of<br/>variances"| FDIST["<strong>F-distribution</strong>(d1, d2)<br/>─────────────<br/>Ratio of Chi-squareds<br/>─────────────<br/>Example: comparing variance<br/>of 2 factory machines<br/>ML: ANOVA, overall regression test"]

    style NORM fill:#4a90d9,stroke:#333,color:#fff
    style MVN fill:#4a90d9,stroke:#333,color:#fff
    style EXP fill:#50c878,stroke:#333,color:#fff
    style LOGN fill:#50c878,stroke:#333,color:#fff
    style CHI fill:#ffa500,stroke:#333,color:#fff
    style TDIST fill:#ffa500,stroke:#333,color:#fff
    style FDIST fill:#ffa500,stroke:#333,color:#fff
    style BETA fill:#9370db,stroke:#333,color:#fff
    style UNIF fill:#9370db,stroke:#333,color:#fff
```

#### Visual Feature Guide: Comparing Continuous Distributions

To help reason about which distribution is suitable, it helps to look at them side-by-side grouped by their support (range) and shape:

![Comparing Continuous Distributions](./continuous_comparisons.png)

<details>
<summary>Python Code for Visualization</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Visual Guide: Comparing Similar Continuous Distributions', fontsize=16, fontweight='bold')

# --- 1. Symmetric Data [-inf, +inf] ---
ax = axes[0, 0]
x = np.linspace(-5, 5, 400)
ax.plot(x, stats.norm.pdf(x), 'k-', linewidth=2.5, label='Normal (Standard)')
ax.plot(x, stats.t.pdf(x, df=2), '--', color='#ff6b6b', linewidth=2.5, label='t-dist (df=2, heavy tails)')
ax.fill_between(x, stats.t.pdf(x, df=2), alpha=0.1, color='#ff6b6b')
ax.set_title('1. Symmetric Data\nNormal vs. t-Distribution', fontsize=12)
ax.legend()

# --- 2. Positive / Right-Skewed Data [0, +inf] ---
ax = axes[0, 1]
x2 = np.linspace(0.01, 5, 400)
ax.plot(x2, stats.expon.pdf(x2, scale=1), color='#4a90d9', linewidth=2.5, label='Exponential (Strict decay)')
ax.plot(x2, stats.lognorm.pdf(x2, s=0.7), color='#ffa500', linewidth=2.5, label='Log-Normal (Peak > 0, long tail)')
ax.plot(x2, stats.gamma.pdf(x2, a=2, scale=0.5), color='#50c878', linewidth=2.5, label='Gamma (Flexible peak)')
ax.set_title('2. Positive & Skewed Data\nExponential vs Log-Normal vs Gamma', fontsize=12)
ax.set_ylim(0, 1.2)
ax.legend()

# --- 3. Bounded Data [0, 1] ---
ax = axes[1, 0]
x3 = np.linspace(0, 1, 400)
ax.plot(x3, stats.uniform.pdf(x3), 'k--', linewidth=2, label='Uniform (No info)')
ax.plot(x3, stats.beta.pdf(x3, 2, 5), color='#9370db', linewidth=2.5, label='Beta(2,5) (Right-skewed constraint)')
ax.fill_between(x3, stats.beta.pdf(x3, 2, 5), alpha=0.2, color='#9370db')
ax.set_title('3. Bounded Data (e.g. Probabilities)\nUniform vs Beta', fontsize=12)
ax.set_ylim(0, 3)
ax.legend()

# --- 4. Test Statistics [0, +inf] ---
ax = axes[1, 1]
x4 = np.linspace(0.01, 8, 400)
ax.plot(x4, stats.chi2.pdf(x4, df=3), color='#ff6b6b', linewidth=2.5, label='Chi-squared (df=3, sum of squares)')
ax.plot(x4, stats.f.pdf(x4, dfn=5, dfd=20), '--', color='#4a90d9', linewidth=2.5, label='F-dist (df1=5, df2=20, ratio of vars)')
ax.set_title('4. Reference Test Statistics\nChi-squared vs F-Distribution', fontsize=12)
ax.set_ylim(0, 0.8)
ax.legend()

plt.tight_layout()
plt.savefig('continuous_comparisons.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

#### Quick-Reference Review: Continuous Distributions

Cover the right columns and try to recall from the name alone.

| Distribution | What does it model? | Key insight / ML Connection | Interview Trigger Phrase |
|---|---|---|---|
| **Normal** | Default for real-valued data | Minimizing MSE $\iff$ Maximizing Normal likelihood | "Central Limit Theorem" or "MSE" |
| **Log-Normal** | Positive, right-skewed data | It's a Normal hiding behind an exponential | "Prices", "salaries", "long tail" |
| **Exponential** | Wait times | It's the continuous memoryless distribution | "Time until event" |
| **Uniform** | Equal probability | Maximum ignorance; used for random search | "Hyperparameter tuning" |
| **Beta** | Probabilities (0 to 1) | Conjugate prior for Binomial (A/B testing) | "Thompson Sampling" or "CTR prior" |
| **Multivariate N.**| Correlated vectors | Covariance matrix eigenvectors $\iff$ PCA | "PCA" or "Gaussian Process" |
| **t-distribution** | Small sample means | Heavier tails = robust to outliers | "regression p-value" or "$n < 30$" |
| **Chi-squared**  | Sum of squared Normals | Used to test independence of categories | "categorical feature selection" |
| **F-distribution** | Ratio of variances | Compares models: "Did these features help?" | "ANOVA" or "nested models" |

---

### Python: Visualizing Continuous Distributions

![Continuous Distributions Gallery](./continuous_distributions_gallery.png)

<details>
<summary>Python Code for Visualization</summary>

```python
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
plt.show()
```

</details>

---

### Real-World Phenomena: Continuous Distributions

To bridge theory and practice, the following visualizes simulated real-world datasets alongside the theoretical continuous distributions that best model them.

![Continuous Phenomena](./continuous_phenomena.png)

<details>
<summary>Python Code for Visualization</summary>

```python
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
plt.show()
```

</details>

#### Why This Distribution and Not Others? (Continuous)

| Phenomenon | Appropriate Distribution | Why this one? | Why not others? |
|---|---|---|---|
| **Human Heights / Exam Scores** | **Normal** | The final value is the additive sum of millions of independent tiny factors (genetics, environment, diet). By the CLT, additive independent effects converge to a Normal distribution. | Not **Log-Normal**, because height isn't purely multiplicative and values cluster symmetrically around the mean. Not **Uniform**, because extreme values are exceedingly rare compared to average values. |
| **Time Until Next Server Crash** | **Exponential** | The probability of crashing in the next hour is constant, regardless of how long the server has been running (memoryless). The data is strictly positive and right-skewed. | Not **Normal**, because times cannot be negative, and the highest probability density is at $t \to 0$ (not a symmetric bell curve). Not **Gamma**, because a single server crash doesn't require waiting for $k$ intermediate events first. |
| **Incomes / House Prices / Stock Prices** | **Log-Normal** | The underlying growth process is *multiplicative* (e.g., getting a 5% raise on your current salary, a stock growing by 2%). When you multiply many positive variables, their logarithm is additive, meaning the log follows a Normal distribution. | Not **Normal**, because a Normal distribution would predict negative house prices/incomes, and doesn't capture the massive right-skew (the "long tail" of billionaires). |
| **Estimating the Probability a Coin is Biased** | **Beta** | The value we are modeling is a *probability* itself, which is strictly bounded between [0, 1]. It serves as a Bayesian prior for binary/binomial events. | Not **Normal**, because a Normal distribution stretches to $\pm \infty$ and wouldn't respect the $[0,1]$ probability boundaries. Not **Uniform**, because after seeing data, we have a "hump" of belief around the most likely probability. |
| **Duration of a Customer Call** | **Gamma** (or Weibull) | Modeling a duration that often has an "activation" phase before failure/completion. Or, waiting for exactly $k$ Poisson events to occur. | Not **Exponential**, because the probability of the call ending in the first 2 seconds is very low (it is NOT memoryless—the chance of hanging up changes as the call progresses). |


---

#### Interview Priority: Continuous Distributions

| What to Know | Priority | Why |
|---|---|---|
| Normal PDF, 68-95-99.7 rule, Z-scores | **Must know** | Foundation of everything |
| Why Normal is everywhere (CLT + max entropy) | **Must know** | Shows deep understanding |
| MSE loss = Normal likelihood | **Must know** | Connects loss functions to distributions |
| t vs Normal: when to use which | **Must know** | Hypothesis testing fundamentals |
| Chi-squared: what it measures, df intuition | **Should know** | Testing, goodness-of-fit |
| F-distribution: ANOVA and regression F-test | **Should know** | Model comparison |
| Exponential: memoryless, Poisson connection | **Should know** | Connects to survival analysis |
| Beta: shape intuition, conjugacy with Binomial | **Must know** | Bayesian A/B testing |
| Multivariate Normal: covariance matrix, Mahalanobis | **Must know** | PCA, GPs, discriminant analysis |
| Log-Normal: when data is multiplicative | **Should know** | Practical modeling judgment |
| Gamma: generalizes Exponential | Nice to have | Completes the picture |
| Dirac delta: empirical distribution connection | Nice to have | MLE theory insight |

---

## 2.2.3 Key Theorems **[H]**

> These three theorems are the "why" behind statistical inference. CLT justifies hypothesis tests, LLN justifies sample estimates, and the Delta Method handles nonlinear functions of statistics.

---

### Central Limit Theorem (CLT)

> The most important theorem in statistics.

**Why learn this**: CLT is literally why A/B testing works. Individual user behavior is binary (Bernoulli), but the sample average is approximately Normal for large $n$ — which lets you compute p-values and confidence intervals. Without CLT, there is no hypothesis testing as we know it.

**Used directly in**:
- **Confidence interval construction**: every CI of the form $\bar{X} \pm z \cdot SE$ works because CLT guarantees $\bar{X}$ is approximately Normal — you use this every time you report an uncertainty estimate
- **Normal approximation to Binomial**: for large $n$, `scipy.stats.norm.cdf()` replaces the slower `scipy.stats.binom.cdf()` — this is how A/B test p-values are computed in practice
- **Mini-batch SGD theory**: the gradient from a mini-batch is an average of per-sample gradients → approximately Normal by CLT → justifies learning rate schedules and convergence guarantees

**Statement**: Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with mean $\mu$ and finite variance $\sigma^2$. Then as $n \to \infty$:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Equivalently: $\bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$ for large $n$.

| Aspect | Details |
|--------|---------|
| **Conditions** | i.i.d., finite mean $\mu$, finite variance $\sigma^2$ |
| **What converges** | The *distribution* of the sample mean (not individual values) |
| **Rate** | $O(1/\sqrt{n})$ — the standard error shrinks as $\sqrt{n}$ |
| **What it does NOT say** | Nothing about individual observations; only about averages |
| **Rule of thumb** | Works well for $n \geq 30$ (less if the original distribution is symmetric) |

> [!IMPORTANT]
> **Why CLT matters for ML/Statistics**:
> 1. **Hypothesis testing**: Test statistics (z-scores, t-scores) are approximately Normal → we can compute p-values
> 2. **Confidence intervals**: $\bar{X} \pm z_{\alpha/2} \cdot \text{SE}$ works because $\bar{X}$ is approximately Normal
> 3. **A/B testing**: Even if conversion events are Bernoulli (very non-Normal), the *average* conversion rate is approximately Normal for large $n$
> 4. **SGD convergence**: Mini-batch gradient estimates are averages → approximately Normal → justifies convergence theory

**Intuition**: Why does averaging produce a Normal shape?

```mermaid
flowchart LR
    subgraph population["Original Population Distribution"]
        I1["Can be ANY shape:<br/>skewed, bimodal,<br/>uniform, etc."]
    end
    
    subgraph averaging["Sampling Distribution of the Mean"]
        A1["n = 1<br/>(Looks like population)"]
        A2["n = 5<br/>(Starting to smooth)"]
        A3["n = 30<br/>(Looks formally Normal!)"]
        A4["n = 100<br/>(Very Normal, tighter)"]
    end
    
    population -->|"Draw samples of size n<br/>and plot their MEANS"| averaging
    A1 --> A2 --> A3 --> A4
    
    style I1 fill:#ff6b6b,stroke:#333,color:#fff
    style A3 fill:#50c878,stroke:#333,color:#fff
    style A4 fill:#4a90d9,stroke:#333,color:#fff
```

**Why it works (intuition, not proof)**: When you average $n$ values, extreme values in one direction tend to cancel with extreme values in the other direction. With more values to average, this cancellation becomes more complete, and what's left is the bell-shaped "residual" — the Normal distribution. The variance shrinks by $1/n$ because with more values, there's more cancellation.

#### Real-World Experiment: The Phenomena vs. The Sample Mean

To make the difference between an individual observation and an average concrete, consider wait times at a coffee shop that follow an **Exponential distribution** with a mean of 5 minutes.

- **The Phenomena (Individual Observations):** If you track 10,000 individual customers, the distribution is extremely right-skewed. Many wait 1 minute, some wait 20 minutes. It looks nothing like a bell curve.
- **The Sample Mean:** If you track 10,000 *days*, and calculate the average wait time of 30 customers each day, the distribution of those daily averages is a incredibly tight **Normal curve** perfectly centered at 5 minutes.

![CLT Individual vs Average](./clt_individual_vs_average.png)

<details>
<summary>Python Code for Experiment</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Simulate 10,000 INDIVIDUAL customers (The Phenomena Distribution)
individual_waits = np.random.exponential(scale=5, size=10000)

# 2. Simulate 10,000 DAYS, taking the AVERAGE of 30 customers per day (Sample Mean Distribution)
daily_averages = [np.random.exponential(scale=5, size=30).mean() for _ in range(10000)]

plt.figure(figsize=(10, 6))

# Plot Individual Observations
plt.hist(individual_waits, bins=50, density=True, alpha=0.5, color='#ff6b6b', 
         label=f'Individual Customers (n=1)\nMean: {individual_waits.mean():.2f}, Variance: {individual_waits.var():.2f}\n(Exponential shape)')

# Plot Sample Means
plt.hist(daily_averages, bins=50, density=True, alpha=0.8, color='#4a90d9', 
         label=f'Daily Averages (n=30)\nMean: {np.mean(daily_averages):.2f}, Variance: {np.var(daily_averages):.2f}\n(Normal shape, tighter!)')

plt.axvline(5, color='black', linestyle='dashed', linewidth=2, label='True Mean (5)')

plt.title('CLT in Action: Individual Phenomena vs. Sample Mean', fontsize=14, fontweight='bold')
plt.xlabel('Wait Time (Minutes)')
plt.ylabel('Density')
plt.xlim(0, 20)
plt.legend()
plt.tight_layout()
plt.savefig('clt_individual_vs_average.png', dpi=150)
plt.show()
```

</details>

#### Python: CLT Convergence Visualization

**In the grid below:**
- **Blue Bars ($\color{#4a90d9}{\blacksquare}$)**: The actual observed empirical data (simulated sample means).
- **Solid Red Line ($\color{#ff6b6b}{—}$)**: The perfect mathematical shape (the theoretical Normal curve). Notice how the messy blue empirical data progressively matches the perfect red theoretical curve as $n$ grows.

![Clt Convergence](./clt_convergence.png)

<details>
<summary>Python Code for Visualization</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.suptitle('Central Limit Theorem: Sample Means Converge to Normal',
             fontsize=16, fontweight='bold')

# Three very non-Normal source distributions
sources = [
    ('Exponential(1)', lambda size: np.random.exponential(1, size)),
    ('Uniform(0,1)', lambda size: np.random.uniform(0, 1, size)),
    ('Bernoulli(0.3)', lambda size: np.random.binomial(1, 0.3, size))
]

sample_sizes = [1, 5, 30, 100]
n_simulations = 10000

for row, (name, sampler) in enumerate(sources):
    for col, n in enumerate(sample_sizes):
        # Simulate n_simulations sample means, each from n observations
        means = [sampler(n).mean() for _ in range(n_simulations)]
        
        ax = axes[row, col]
        ax.hist(means, bins=50, density=True, alpha=0.7, color='#4a90d9', edgecolor='black')
        
        # Overlay theoretical Normal
        mu = np.mean(means)
        sigma = np.std(means)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
        if col == 0:
            ax.set_ylabel(name, fontsize=12, fontweight='bold')
        if row == 0:
            ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
        if row == 0 and col == 3:
            ax.legend()

plt.tight_layout()
plt.savefig('clt_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. A/B Testing (Agenda 1.2.1):**
> CLT is literally why A/B testing works. Even though individual user behavior is binary (convert or not — a Bernoulli), the sample *proportion* $\hat{p} = \bar{X}$ is approximately Normal for large $n$: $\hat{p} \sim \mathcal{N}(p, p(1-p)/n)$. This lets us use z-tests to compare conversion rates.
>
> **2. Mini-batch SGD (Agenda 2.6.3):**
> The gradient computed from a mini-batch is an average of per-sample gradients. By CLT, this average is approximately Normal around the true full-batch gradient, with variance decreasing as $1/\text{batch\_size}$. This is why larger batches give more stable (but more expensive) gradient estimates.
>
> **3. Normal Approximation to Binomial:**
> For $n$ large, $\text{Binomial}(n, p) \approx \mathcal{N}(np, np(1-p))$. This is a direct consequence of CLT applied to a sum of Bernoullis.

---

### Law of Large Numbers

**Why learn this**: LLN justifies using training loss as a proxy for true risk. It's why Monte Carlo estimation works, why more data gives better models, and why sample averages converge to population means. It's the theoretical foundation of empirical risk minimization — the basis of all supervised ML.

**Used directly in**:
- **Monte Carlo estimation**: approximate $E[f(X)]$ by sampling: `np.mean([f(x) for x in samples])` — LLN guarantees this converges to the true expectation. Used in Bayesian inference (MCMC), reinforcement learning (policy evaluation), and option pricing
- **Empirical risk minimization**: training loss $\frac{1}{n}\sum L(f(x_i), y_i) \to E[L]$ as $n \to \infty$ — LLN is why "more data = better model" is true
- **Cross-validation reliability**: the average CV score converges to the true expected performance by LLN — more folds = more reliable estimate

**Statement**: Let $X_1, X_2, \ldots, X_n$ be i.i.d. with mean $\mu$. Then:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{} \mu \quad \text{as } n \to \infty$$

| Version | Convergence Type | Statement |
|---------|-----------------|-----------|
| **Weak LLN** | In probability | $\forall \epsilon > 0: P(\|\bar{X}_n - \mu\| > \epsilon) \to 0$ |
| **Strong LLN** | Almost surely | $P(\lim_{n\to\infty} \bar{X}_n = \mu) = 1$ |

> [!TIP]
> **CLT vs LLN — know the difference**:
>
> | | Law of Large Numbers | Central Limit Theorem |
> |---|---|---|
> | **Says** | $\bar{X}_n \to \mu$ (converges to a number) | $\bar{X}_n \sim \mathcal{N}(\mu, \sigma^2/n)$ (has a specific shape) |
> | **Tells you** | The sample mean *works* as an estimator | *How* the sample mean fluctuates |
> | **Cares about** | Whether $\bar{X}_n$ hits $\mu$ | The distribution *around* $\mu$ |
> | **Useful for** | Justifying sample estimates | Computing confidence intervals and p-values |

#### Python: LLN Convergence Visualization

![Lln Convergence](./lln_convergence.png)

<details>
<summary>Python Code for Visualization</summary>

```python
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
plt.show()
```

</details>

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. Empirical Risk Minimization (Agenda 3.1):**
> LLN justifies why we can use the *training loss* (average loss over $n$ samples) as a proxy for the *true risk* (expected loss over the population). As $n \to \infty$, $\frac{1}{n}\sum L(f(x_i), y_i) \to E[L(f(X), Y)]$. Without LLN, ML would have no theoretical foundation.
>
> **2. Monte Carlo Methods (Agenda 2.4.3):**
> MCMC sampling works because of LLN: the average of samples from a distribution converges to the true expectation. More samples = better approximation.

---

### Delta Method

Approximates the distribution of a *function* of a random variable.

**Why learn this**: Many business metrics are ratios: revenue per user, clicks per impression, cost per acquisition. You can't just use the standard SE formula for these. The Delta Method gives you analytical standard errors for ratio metrics — critical for A/B testing at scale where bootstrapping is too slow.

> [!IMPORTANT]
> **Common Interview Question: If the CLT says the sample mean converges to a Normal distribution, why do we need the Delta Method at all?**
>
> **The Answer:** You are 100% correct that the CLT guarantees your ratio metric ($R = \bar{X}/\bar{Y}$) will be approximately Normal when the sample size is large. 
> 
> However, to actually run an A/B test (like a Z-test or t-test), knowing the *shape* is Normal isn't enough. You must calculate the **Standard Error (variance)** of that Normal curve to get a p-value: $Z = \frac{R_{\text{treat}} - R_{\text{control}}}{\sqrt{SE_{\text{treat}}^2 + SE_{\text{control}}^2}}$.
>
> The CLT doesn't magically provide the formula for the variance of a ratio. You absolutely cannot just divide $\text{Var}(X) / \text{Var}(Y)$. The **Delta Method** is the mathematical tool used to *approximate the variance* of a function of random variables so that you can plug it into your CLT-justified hypothesis test!

**Used directly in**:
- **Ratio metric SEs in A/B testing**: revenue per user = total_revenue / total_users is a ratio — the Delta Method gives you $SE(\hat{R})$ analytically without bootstrapping, which is critical at scale (millions of users)
- **Variance-stabilizing transforms**: for Poisson data, $g(X) = \sqrt{X}$ stabilizes variance (since $\text{Var}(X) = \mu$ varies) — the Delta Method tells you which transform works
- **Confidence intervals for nonlinear functions**: CIs for $\log(\hat{p})$, $1/\hat{\mu}$, or any smooth function of an estimator — the Delta Method provides the SE without simulation

**Statement**: If $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$, then for a differentiable function $g$:

$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2 \sigma^2)$$

**Practical form**: 

$$\text{Var}(g(\bar{X})) \approx [g'(\mu)]^2 \cdot \text{Var}(\bar{X})$$

<details>
<summary><strong>Worked Example: Variance of a Ratio Metric in A/B Testing</strong></summary>

**Setup**: You want to test whether revenue per user ($R = \text{total\_revenue} / \text{total\_users}$) differs between treatment and control.

This is a *ratio* of two random variables — you can't just use the standard formula for SE of a mean.

**Delta Method approach**:
Let $g(x, y) = x/y$ (revenue / users). The gradient is:

$$\nabla g = \left(\frac{1}{y}, -\frac{x}{y^2}\right)$$

By the multivariate Delta Method:

$$\text{Var}\left(\frac{\bar{X}}{\bar{Y}}\right) \approx \frac{1}{\mu_Y^2}\text{Var}(\bar{X}) + \frac{\mu_X^2}{\mu_Y^4}\text{Var}(\bar{Y}) - \frac{2\mu_X}{\mu_Y^3}\text{Cov}(\bar{X}, \bar{Y})$$

```python
import numpy as np

np.random.seed(42)
n = 10000

# Simulate: each user has a revenue (possibly 0 if they don't buy)
p_buy = 0.1
revenue_if_buy = np.random.lognormal(mean=3, sigma=1, size=n)
bought = np.random.binomial(1, p_buy, size=n)
revenue = revenue_if_buy * bought

# Ratio metric: revenue per user
total_rev = revenue.sum()
total_users = n
ratio = total_rev / total_users

# Bootstrap SE (truth)
bootstrap_ratios = []
for _ in range(5000):
    idx = np.random.choice(n, n, replace=True)
    bootstrap_ratios.append(revenue[idx].mean())
bootstrap_se = np.std(bootstrap_ratios)

# Direct SE (wrong — ignores ratio structure)
direct_se = revenue.std() / np.sqrt(n)

print(f"Revenue per user: {ratio:.2f}")
print(f"Bootstrap SE:     {bootstrap_se:.4f}")
print(f"Direct SE:        {direct_se:.4f}")
print(f"They should be similar here since denominator is fixed (n)")
```

**Key takeaway**: The Delta Method gives you analytical standard errors for complex metrics (ratios, percentages, log-transformed values) without needing to bootstrap — much faster for large-scale A/B testing platforms.

</details>

> [!NOTE]
> **Fundamental ML Connections**
>
> **1. A/B Testing Ratio Metrics (Agenda 1.2.1):**
> Many business metrics are ratios: revenue per user, clicks per impression, time per session. The Delta Method provides the standard error needed for hypothesis testing on these metrics without resorting to expensive bootstrapping.
>
> **2. Variance Stabilizing Transformations:**
> The Delta Method can be used "in reverse" — find a transformation $g$ such that $\text{Var}(g(X))$ is approximately constant. For Poisson data, the square root transform $g(X) = \sqrt{X}$ stabilizes variance (since $\text{Var}(X) = \mu$ varies).

---

#### Interview Priority: Key Theorems

| What to Know | Priority | Why |
|---|---|---|
| CLT statement and conditions | **Must know** | Foundation of all statistical testing |
| CLT intuition: why averaging produces Normality | **Must know** | Shows deep understanding |
| CLT application to A/B testing | **Must know** | Most common applied context |
| LLN: sample averages converge to true mean | **Must know** | Justifies empirical risk minimization |
| CLT vs LLN: what each says | **Should know** | Common interview confusion point |
| Delta Method: variance of g(X) | **Should know** | Ratio metrics in A/B testing |
| Normal approximation to Binomial | **Should know** | Direct CLT application |

---

## 2.2.4 Distribution Relationships **[M]**

> Understanding how distributions relate to each other transforms a list of formulas into a connected web of intuition.

---

### Exponential Family Unification

Most distributions we've covered belong to the **exponential family** — a powerful unification.

**Why learn this**: Exponential family is the reason GLMs work, the reason conjugate priors exist, and the reason MLE has nice properties. Understanding it transforms the distribution zoo from a list of formulas into a single unified framework — and shows interviewers you see the deep structure.

**Used directly in**:
- **GLM model selection**: knowing which distribution is exponential family tells you which GLM to use — Normal → OLS, Bernoulli → logistic, Poisson → Poisson regression, Gamma → Gamma regression
- **Sufficient statistics**: for exponential family, MLE reduces to matching model moments to sample moments — this is why `np.mean(x)` is the MLE for Poisson $\lambda$ and Normal $\mu$
- **Conjugate prior lookup**: every exponential family has a conjugate prior → closed-form Bayesian updates without MCMC. Beta-Binomial, Gamma-Poisson, Normal-Normal are the three you'll use most

**General form**:

$$p(x \mid \boldsymbol{\eta}) = h(x) \exp(\boldsymbol{\eta}^T \mathbf{T}(x) - A(\boldsymbol{\eta}))$$

| Term | Name | Role |
|------|------|------|
| $\boldsymbol{\eta}$ | Natural parameter | The "canonical" parameterization |
| $\mathbf{T}(x)$ | Sufficient statistic | All the information $x$ carries about $\boldsymbol{\eta}$ |
| $A(\boldsymbol{\eta})$ | Log-partition function | Normalizing constant (generates moments!) |
| $h(x)$ | Base measure | Distribution "skeleton" |

**Which distributions are exponential family?**

| Distribution | Natural Parameter $\eta$ | Sufficient Statistic $T(x)$ |
|---|---|---|
| Bernoulli$(p)$ | $\log(p/(1-p))$ (log-odds!) | $x$ |
| Binomial$(n, p)$ | $\log(p/(1-p))$ | $x$ |
| Poisson$(\lambda)$ | $\log(\lambda)$ | $x$ |
| Normal$(\mu, \sigma^2)$ | $(\mu/\sigma^2, -1/2\sigma^2)$ | $(x, x^2)$ |
| Exponential$(\lambda)$ | $-\lambda$ | $x$ |
| Gamma$(\alpha, \beta)$ | $(\alpha - 1, -\beta)$ | $(\log x, x)$ |
| Beta$(\alpha, \beta)$ | $(\alpha - 1, \beta - 1)$ | $(\log x, \log(1-x))$ |

**Not exponential family**: Uniform (support depends on parameter), Student's t (no sufficient statistic of fixed dimension).

> [!IMPORTANT]
> **Why exponential family matters for ML**:
> 1. **GLMs**: Generalized Linear Models work *because* the response distribution is exponential family. The link function connects the natural parameter to the linear predictor $X\beta$.
> 2. **Conjugate priors**: Every exponential family distribution has a conjugate prior — enabling closed-form Bayesian updates.
> 3. **MLE properties**: MLE for exponential family always exists, is unique, and the sufficient statistic is all you need. The MLE equations reduce to "match the model's expected sufficient statistics to the observed sufficient statistics."
> 4. **Maximum entropy**: The exponential family distribution is the *maximum entropy* distribution that matches given moment constraints.

---

### Key Relationships Diagram

**Why learn this**: Interviewers love asking "how does X relate to Y?" for distributions. Being able to sketch the relationship diagram from memory — showing how Bernoulli builds to Binomial builds to Normal via CLT, how Exponential specializes Gamma which becomes Chi-squared — demonstrates mastery rather than memorization.

**Used directly in**:
- **Choosing the right statistical test**: Bernoulli → Binomial → Normal (CLT) is the chain that takes you from individual binary events to z-tests for proportions — knowing this chain means you can derive the right test from first principles
- **Debugging model assumptions**: if residuals look Chi-squared (right-skewed, positive), your data might need a Gamma or Poisson model instead of Normal — the relationship diagram tells you where to look
- **Interview derivations**: "Derive the distribution of the sample variance" requires knowing Normal → Chi-squared. "Why does the t-test use the t-distribution?" requires knowing Normal/Chi-squared → t. These chains are the building blocks

```mermaid
flowchart TD
    BERN["Bernoulli(p)<br/>Single trial"]
    BINOM["Binomial(n, p)<br/>n trials"]
    POIS["Poisson(lambda)<br/>Counts/time"]
    GEOM["Geometric(p)<br/>Until 1st success"]
    NEGBIN["Neg. Binomial(r,p)<br/>Until r-th success"]
    MULTI["Multinomial(n, p)<br/>K categories"]
    
    NORM["Normal(mu, s^2)<br/>The default"]
    EXP["Exponential(lambda)<br/>Wait times"]
    GAMMA["Gamma(a, b)<br/>Positive values"]
    CHI["Chi-squared(k)<br/>Sum of Z^2"]
    TDIST["t(k)<br/>Small samples"]
    FDIST["F(d1, d2)<br/>Variance ratios"]
    BETA["Beta(a, b)<br/>Probabilities"]
    LOGN["Log-Normal<br/>Multiplicative"]
    
    %% Discrete relationships
    BERN -->|"Sum of n iid"| BINOM
    BERN -->|"K categories"| MULTI
    BINOM -->|"n large, p small<br/>lambda = np"| POIS
    BINOM -->|"n large<br/>(CLT)"| NORM
    GEOM -->|"r = 1<br/>special case"| NEGBIN
    
    %% Continuous relationships
    EXP -->|"alpha = 1"| GAMMA
    GAMMA -->|"a=k/2, b=1/2"| CHI
    NORM -->|"Square iid<br/>sum Z_i^2"| CHI
    NORM -->|"Z / sqrt(V/k)"| TDIST
    CHI -->|"Ratio<br/>(U/d1)/(V/d2)"| FDIST
    TDIST -->|"T^2"| FDIST
    NORM -->|"exp(X)"| LOGN
    
    %% Cross relationships
    POIS -.->|"Inter-arrival<br/>times"| EXP
    BETA -.->|"Conjugate prior<br/>for Bernoulli"| BERN
    TDIST -.->|"df -> inf"| NORM
    
    %% Styling
    style BERN fill:#4a90d9,stroke:#333,color:#fff
    style BINOM fill:#4a90d9,stroke:#333,color:#fff
    style POIS fill:#4a90d9,stroke:#333,color:#fff
    style MULTI fill:#4a90d9,stroke:#333,color:#fff
    style GEOM fill:#4a90d9,stroke:#333,color:#fff
    style NEGBIN fill:#4a90d9,stroke:#333,color:#fff
    style NORM fill:#50c878,stroke:#333,color:#fff
    style EXP fill:#50c878,stroke:#333,color:#fff
    style GAMMA fill:#50c878,stroke:#333,color:#fff
    style CHI fill:#ffa500,stroke:#333,color:#fff
    style TDIST fill:#ffa500,stroke:#333,color:#fff
    style FDIST fill:#ffa500,stroke:#333,color:#fff
    style BETA fill:#9370db,stroke:#333,color:#fff
    style LOGN fill:#50c878,stroke:#333,color:#fff
```

#### Key Transformation Chains

| Chain | Relationship | Why It Matters |
|-------|-------------|----------------|
| Bernoulli $\to$ Binomial $\to$ Normal | Sum $\to$ CLT | A/B testing: binary events $\to$ Normal test statistics |
| Exponential $\to$ Gamma $\to$ Chi-squared | Special cases | Hypothesis testing distribution family |
| Normal $\to$ Chi-squared $\to$ t $\to$ F | Squared $\to$ ratio | The complete testing distribution hierarchy |
| Poisson $\leftrightarrow$ Exponential | Counts $\leftrightarrow$ times | Two views of the same process |
| Beta $\to$ Bernoulli/Binomial | Prior $\to$ likelihood | Bayesian conjugacy |
| Normal $\to$ Log-Normal | $\exp(\cdot)$ | Multiplicative vs additive processes |

---

#### Interview Priority: Distribution Relationships

| What to Know | Priority | Why |
|---|---|---|
| Bernoulli $\to$ Binomial $\to$ Normal (CLT) | **Must know** | Core testing pipeline |
| Poisson $\leftrightarrow$ Exponential (counts vs times) | **Must know** | Common interview question |
| Normal $\to$ Chi-squared $\to$ t $\to$ F chain | **Should know** | Testing distribution hierarchy |
| Beta as conjugate prior for Bernoulli | **Must know** | Bayesian A/B testing |
| Exponential family concept | **Should know** | GLM foundation |
| Full relationship diagram from memory | **Should know** | Section deliverable |

---

## Connections Map

How Section 2.2 connects to the rest of the study plan:

```mermaid
flowchart LR
    SEC22["Section 2.2<br/>Distributions"]
    
    SEC21["2.1 Probability<br/>Basics"]
    SEC23["2.3 Statistical<br/>Inference"]
    SEC25["2.5 Linear<br/>Algebra"]
    SEC26["2.6 Calculus &<br/>Optimization"]
    SEC27["2.7 Information<br/>Theory"]
    SEC31["3.1 ML<br/>Basics"]
    
    SEC21 -->|"Bayes feeds into<br/>Beta-Binomial"| SEC22
    SEC22 -->|"CLT enables<br/>hypothesis tests"| SEC23
    SEC22 -->|"MVN uses<br/>cov matrices"| SEC25
    SEC22 -->|"Normal PDF is<br/>quadratic for MLE"| SEC26
    SEC22 -->|"Cross-entropy from<br/>Categorical dist"| SEC27
    SEC22 -->|"Distributions define<br/>loss functions"| SEC31
    
    style SEC22 fill:#4a90d9,stroke:#333,color:#fff
```

| This Section | Connects To | How |
|---|---|---|
| Bernoulli/Binomial | 2.3 Hypothesis Testing | z-tests, p-values |
| Normal | 2.3 Confidence Intervals | CI = $\bar{X} \pm z \cdot \text{SE}$ |
| Multivariate Normal | 2.5 Linear Algebra | Covariance matrix, eigendecomposition |
| All distributions | 2.6 Optimization | MLE = finding optimal parameters |
| Categorical/Normal | 2.7 Information Theory | Cross-entropy, KL divergence |
| All distributions | 3.1 ML Basics | Loss functions = negative log-likelihoods |
| Beta distribution | 2.1.4 Conjugate Priors | Beta-Binomial conjugacy |
| CLT | 2.3.2 Hypothesis Testing | Justifies Normal-based tests |

---

## Interview Cheat Sheet

**Quick-fire answers for common distribution questions:**

| Question | Answer |
|----------|--------|
| "When Poisson vs Binomial?" | Poisson: rare events in continuous interval, unbounded counts. Binomial: fixed $n$ trials, bounded count. |
| "When t vs Normal?" | t-test when $\sigma$ unknown AND $n$ small. For large $n$, they're equivalent. |
| "What's special about the Normal?" | CLT + max entropy + math convenience. MSE loss = Normal likelihood. |
| "Explain CLT in one sentence" | Sample averages become Normal regardless of the original distribution, given enough samples. |
| "What does mean = variance mean?" | Poisson fingerprint. If variance >> mean, use Negative Binomial instead. |
| "Why Beta for A/B testing?" | Conjugate prior for Bernoulli — posterior updates are closed-form, no MCMC needed. |
| "What is Mahalanobis distance?" | Euclidean distance adjusted for correlations via the inverse covariance matrix. |
| "Name three properties of Normal" | Symmetric, sum of Normals is Normal, fully defined by mean and variance. |
| "Why does chi-squared divide by n-1?" | Because estimating the mean "uses up" 1 degree of freedom. |
| "What's the Delta Method?" | Approximates variance of $g(\bar{X})$ as $[g'(\mu)]^2 \cdot \text{Var}(\bar{X})$. Used for ratio metrics. |

---

## Learning Objectives Checklist

### Section 2.2 Deliverables

- [ ] **Distribution relationship diagram**: Draw the full relationship diagram from memory (Bernoulli $\to$ Binomial $\to$ Normal, Exponential $\to$ Gamma $\to$ Chi-squared, etc.)
- [ ] **Distribution summary**: For each distribution, write from memory: PMF/PDF, mean, variance, and one real-world use case
- [ ] **CLT convergence**: Run and understand the CLT convergence visualization code — explain why it works
- [ ] **Quick identification**: Given a data description, identify the appropriate distribution in under 30 seconds (use the decision flowcharts)
- [ ] **Interview problems**: Solve 5 distribution-selection problems without notes (e.g., "A website gets 3 errors/hour on average. What's the probability of 0 errors in the next hour?")
- [ ] **Exponential family**: List 5 distributions that are exponential family and explain why it matters for GLMs
- [ ] **Whiteboard**: Derive the Normal approximation to the Binomial using CLT

### Self-Quiz Questions

1. Your A/B test has binary outcomes. What distribution models the total conversions? What distribution approximates it for large $n$?
2. You're modeling insurance claims (rare events, overdispersed). Poisson or Negative Binomial? Why?
3. Sketch the Beta distribution for $\text{Beta}(1,1)$, $\text{Beta}(5,5)$, and $\text{Beta}(2,8)$. What does each represent as a prior?
4. What's the difference between CLT and LLN? Give a one-sentence answer for each.
5. Your colleague says "the data is Normal" about a column of user session durations. Is this likely? What distribution would you expect?

---

> [!TIP]
> **Study order within 2.2**: Start with 2.2.1 (discrete) → 2.2.2 (continuous, focus on Normal first) → 2.2.3 (CLT/LLN) → 2.2.4 (relationships). Save the relationship diagram for last — it ties everything together.

