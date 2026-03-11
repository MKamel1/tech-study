# Causal Inference Applied Scientist: Comprehensive Gap Analysis

---
## Document Outline
- [Why Multi-Armed Bandits Were Skipped](#why-multi-armed-bandits-were-skipped)
- [Causal Inference Gap Analysis](#causal-inference-gap-analysis)
- [Big Tech Minimum Bar for Causal AS](#big-tech-minimum-bar-for-causal-as)
- [Recommended Additions to Agenda](#recommended-additions-to-agenda)
---

## Executive Summary
This document provides a comprehensive gap analysis of the current AS study agenda against big tech hiring requirements for causal inference roles. It identifies missing topics, explains the multi-armed bandit oversight, and defines the minimum competency bar across causal depth, experimentation, ML breadth, and soft skills.

---

## Why Multi-Armed Bandits Were Skipped

> [!IMPORTANT]
> Multi-armed bandits (MAB) are **at the intersection of experimentation and decision-making** — a critical gap in the current agenda.

### The Oversight Explained

| Current Coverage | What's Missing |
|------------------|----------------|
| Section 9.2 RL marked as **[L]** | MAB framed as "RL basics" but it's much more relevant |
| No dedicated MAB section | Bandits are NOT pure RL — they're practical experimentation |
| No contextual bandits | This is the production-relevant variant |

### Why Bandits Matter for Causal-Focused AS

1. **Bandits ARE causal inference** — they solve the explore-exploit tradeoff causally
2. **Production experimentation platforms** at Amazon, Netflix, Meta use bandits
3. **Bridges pure experimentation and adaptive policies**
4. **Common interview topic** — "How would you run experiments with limited traffic?"

### How Bandits Connect to Causal Topics

```
Traditional A/B Testing (Static)
        │
        ▼
   Bandits (Adaptive)
        │
   ┌────┴────┐
   ▼         ▼
Thompson    UCB
Sampling    
   │
   ▼
Contextual Bandits
(Personalized Treatment Assignment)
   │
   ▼
Off-Policy Evaluation
(Counterfactual Learning)
   │
   ▼
Causal Inference Connection:
- IPW for OPE
- Doubly Robust estimators
- Treatment effect heterogeneity
```

### Recommended Addition

Add Section 1.7 or integrate into 1.2.1:

```markdown
### 1.7 Adaptive Experimentation & Bandits **[H]**
- [ ] 1.7.1 Multi-Armed Bandits **[H]**
  - Explore-exploit tradeoff **[H]**
  - Thompson Sampling (Bayesian approach) **[H]**
  - UCB (frequentist approach) **[M]**
  - When bandits vs A/B tests **[C]**
- [ ] 1.7.2 Contextual Bandits **[H]**
  - Policy learning connection **[H]**
  - LinUCB, Thompson for contextual **[M]**
- [ ] 1.7.3 Off-Policy Evaluation **[M]**
  - IPW for counterfactual estimates **[H]**
  - Doubly robust OPE **[M]**
  - Logging policy requirements **[M]**
```

---

## Causal Inference Gap Analysis

### Current Agenda vs Required Coverage

| Topic | Current Status | Required for Causal AS | Gap Severity |
|-------|----------------|------------------------|--------------|
| **Potential Outcomes** | ✅ Covered [C] | Must master | None |
| **SCM/DAGs** | ✅ Covered [C] | Must master | None |
| **DiD** | ✅ Covered [C] | Must master | None |
| **RDD** | ✅ Covered [H] | Strong competence | None |
| **IV** | ✅ Covered [C] | Must master | None |
| **Synthetic Control** | ✅ Covered [C] | Must master | None |
| **Propensity Score Methods** | ❌ Missing | Must master | **CRITICAL** |
| **Matching** | ❌ Missing | Strong competence | **HIGH** |
| **CUPED/Variance Reduction** | ❌ Missing | Must master | **CRITICAL** |
| **Multi-Armed Bandits** | ❌ Missing | Strong competence | **HIGH** |
| **Interference/Network Effects** | ❌ Missing | Must know | **MEDIUM** |
| **Sequential Testing** | ❌ Missing | Should know | **MEDIUM** |
| **Causal Forests** | ✅ Covered [H] | Strong competence | None |
| **Meta-learners** | ✅ Covered [H] | Strong competence | None |
| **Sensitivity Analysis** | ⚠️ Under-prioritized [M] | Strong competence | **MEDIUM** |
| **Mediation Analysis** | ✅ Covered [M] | Know concepts | None |
| **Regression Adjustment** | ❌ Missing | Should know | **MEDIUM** |
| **Cluster Randomization** | ❌ Missing | Should know | **MEDIUM** |

---

### CRITICAL GAPS (Must Fix)

#### Gap 1: Propensity Score Methods

> [!CAUTION]
> This is the single largest gap. Every causal interview covers propensity scores.

**What's needed:**

| Subtopic | Importance | What You Must Know |
|----------|------------|-------------------|
| Propensity Score Matching | **[C]** | Nearest neighbor, caliper, greedy vs optimal |
| Inverse Probability Weighting (IPW) | **[C]** | Horvitz-Thompson, normalized weights |
| Augmented IPW (AIPW) | **[H]** | Doubly robust property |
| Propensity score as covariate | **[H]** | Stratification, regression adjustment |
| Diagnostics | **[C]** | Balance checks, overlap, standardized differences |
| Positivity violations | **[C]** | Trimming, truncation, extrapolation risks |

**Interview question examples:**
- "Walk me through how you'd estimate ATT with observational data"
- "Your propensity scores have extreme values (0.01, 0.99). What do you do?"
- "Why is matching on propensity score better than matching on covariates?"

---

#### Gap 2: Variance Reduction in Experiments

> [!CAUTION]
> Every experimentation platform team expects this knowledge.

**What's needed:**

| Subtopic | Importance | Core Concepts |
|----------|------------|---------------|
| CUPED | **[C]** | Pre-experiment covariate adjustment |
| Stratified randomization | **[H]** | Block randomization by key dimensions |
| Poststratification | **[H]** | Adjusting for known population proportions |
| Regression adjustment | **[H]** | Lin estimator, covariate-adjusted ATE |
| Variance-weighted metrics | **[M]** | Delta method for ratio metrics |

**Why this matters:**
- Reduces required sample size by 20-50%
- Expected knowledge at Amazon, Meta, LinkedIn, Airbnb
- Shows you understand production experimentation

---

### HIGH PRIORITY GAPS

#### Gap 3: Matching Methods

| Method | Priority | Coverage Needed |
|--------|----------|-----------------|
| Exact matching | **[H]** | Coarsened exact matching (CEM) |
| Nearest neighbor matching | **[H]** | With/without replacement, calipers |
| Mahalanobis matching | **[M]** | Multivariate distance |
| Genetic matching | **[L]** | Conceptual awareness |

---

#### Gap 4: Multi-Armed Bandits (Detailed Above)

Key areas:
- Thompson Sampling (Bayesian) **[H]**
- UCB (frequentist) **[M]**
- Contextual bandits **[H]**
- Off-policy evaluation **[M]**

---

#### Gap 5: Network Effects & Interference

| Subtopic | Importance | Why It Matters |
|----------|------------|----------------|
| SUTVA violations | **[H]** | When standard methods fail |
| Cluster randomization | **[H]** | Solution for interference |
| Switchback designs | **[M]** | Marketplace/ride-share experiments |
| Network exposure mapping | **[L]** | Social network experiments |

---

### MEDIUM PRIORITY GAPS

#### Gap 6: Sequential Testing & Always-Valid Inference

- Group sequential methods **[M]**
- Alpha spending functions **[L]**
- Confidence sequences **[M]**
- When to use vs fixed-horizon **[H]**

#### Gap 7: Regression Adjustment for RCTs

- Lin (2013) estimator **[H]**
- When regression helps vs hurts **[H]**
- ANCOVA for pre-post designs **[M]**

---

## Big Tech Minimum Bar for Causal AS

### The Hiring Bar Framework

Big tech causal AS roles evaluate across **4 dimensions**:

```
┌─────────────────────────────────────────────────┐
│             CAUSAL AS HIRING BAR                │
├─────────────┬─────────────┬─────────────────────┤
│   CAUSAL    │    ML       │  EXPERIMENTATION    │
│   DEPTH     │  BREADTH    │    FLUENCY          │
│    40%      │    25%      │      25%            │
├─────────────┴─────────────┴─────────────────────┤
│           COMMUNICATION & PRODUCT               │
│                    10%                          │
└─────────────────────────────────────────────────┘
```

---

### Dimension 1: Causal Depth (40% of evaluation)

**Must demonstrate:**

| Competency | Bar | How They Test |
|------------|-----|---------------|
| Framework fluency | Can explain both PO and SCM, when each is useful | "Explain identification in DAG terms AND Rubin terms" |
| Method selection | Know which method for which scenario | "RDD fails here. What else?" |
| Assumption checking | Can critique designs for assumption violations | "What could go wrong with this DiD?" |
| Modern causal ML | Can implement meta-learners, understand DML | "Compare X-learner vs DR-learner" |
| Tools | DoWhy proficiency, EconML familiarity | Live coding with these libraries |

**Typical depth questions:**
1. "Draw the DAG for this problem. What do you need to control for?"
2. "Derive the IV estimator. What happens under heterogeneous effects?"
3. "Your parallel trends assumption is violated. Options?"
4. "We have 1M users. Some are never treated. How do you estimate CATE?"

---

### Dimension 2: ML Breadth (25% of evaluation)

> [!IMPORTANT]
> **Senior** Causal AS roles have a HIGHER breadth bar. You're expected to own end-to-end solutions, mentor others, and pass general ML interview rounds even for causal-focused positions.

#### Why Breadth Matters More for Senior

| Senior Expectation | Why It Requires Breadth |
|-------------------|-------------------------|
| Own full pipeline | Causal is one piece — you also need data, features, models, deployment |
| Cross-functional leadership | PMs ask about ML, not just causal |
| Mentorship | Junior AS ask general questions |
| Fallback solutions | "If causal fails, what's the ML alternative?" |
| System design rounds | Must design complete ML systems |

---

#### Breadth Area 1: Supervised Learning **[MUST MASTER]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Bias-variance tradeoff | Explain intuitively to non-technical | ✅ [C] | None |
| Regularization (L1/L2/ElasticNet) | Know when each, derive gradients | ✅ [C] | None |
| Tree ensembles (RF, XGBoost, LightGBM) | Tune confidently, explain internals | ✅ [C] | None |
| **Imbalanced learning** | SMOTE, class weights, threshold tuning, AUPRC | ⚠️ Brief in 6.3.3 | **HIGH GAP** |
| **Model calibration** | Platt scaling, isotonic, reliability diagrams | ⚠️ [M] only | **MEDIUM GAP** |
| **Interpretability (SHAP, LIME)** | Explain to stakeholders, feature importance | ❌ Missing | **HIGH GAP** |
| Cross-validation | Time-series CV, grouped CV, leakage prevention | ✅ [H] | None |

**Interview questions you must handle:**
- "Your model has 0.95 AUC but 0.02 precision. Explain and fix."
- "Walk me through how XGBoost builds trees. What's the objective?"
- "Show me SHAP values for this prediction. What drives it?"

---

#### Breadth Area 2: Feature Engineering **[STRONG COMPETENCE]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Numerical transforms | Log, Box-Cox, binning, outlier handling | ✅ [H] | None |
| Categorical encoding | Target encoding, embeddings, high-cardinality | ✅ [M] | Minor |
| **Feature selection** | Filter, wrapper, embedded methods | ⚠️ Implicit | **MEDIUM GAP** |
| **Feature stores** | Offline vs online, point-in-time correctness | ✅ [M] | None |
| Time-based features | Lags, rolling stats, seasonality indicators | ✅ [H] | None |

---

#### Breadth Area 3: Deep Learning **[MODERATE COMPETENCE]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Backpropagation | Derive on whiteboard | ✅ [H] | None |
| Activation functions | Know when to use each, vanishing gradient | ✅ [H] | None |
| Batch norm / Dropout | Explain WHY they work | ✅ [M] | None |
| **Transformers/Attention** | Explain architecture, compute complexity | ✅ [H] | None |
| **Embeddings** | Use pretrained, fine-tuning decisions | ⚠️ [M] | Minor |
| **Transfer learning (non-LLM)** | When to freeze layers, domain adaptation | ❌ Missing | **MEDIUM GAP** |

---

#### Breadth Area 4: Time Series & Forecasting **[STRONG FOR SUPPLY CHAIN]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Stationarity & differencing | ADF/KPSS interpretation | ✅ [C] | None |
| ARIMA/SARIMA | Model selection, ACF/PACF | ✅ [H] | None |
| Prophet / ML forecasting | Know trade-offs, when to use | ✅ [H] | None |
| **Forecast reconciliation** | Hierarchical, top-down, bottom-up | ✅ [M] | None |
| **Evaluation pitfalls** | Time leakage, proper backtesting | ⚠️ Implicit | Minor |

---

#### Breadth Area 5: NLP/LLM **[MODERATE COMPETENCE]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Text embeddings | Use sentence-transformers | ✅ [H] | None |
| Transformers architecture | Attention mechanism, encoder/decoder | ✅ [H] | None |
| RAG basics | Chunking, retrieval, generation | ✅ [H] | None |
| Prompt engineering | Few-shot, chain-of-thought | ✅ [H] | None |
| **Evaluation** | BLEU, ROUGE, human eval design | ⚠️ [H] | Minor |

---

#### Breadth Area 6: Statistics **[MUST MASTER]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Bayes' theorem | Common applications, MLE vs MAP | ✅ [C] | None |
| CLT / LLN | Intuition and when they apply | ✅ [C]/[H] | None |
| Hypothesis testing | Type I/II, power, p-value interpretation | ✅ [C] | None |
| **Bootstrap** | When to use, implementation | ⚠️ [M] in 4.5 | Minor |
| **Permutation tests** | Implementation, when preferred | ✅ [M] | None |
| **Bayesian basics** | Priors, posteriors, conjugacy | ✅ [M] | None |

---

#### Breadth Area 7: System Design **[REQUIRED FOR SENIOR]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Batch vs real-time | Trade-offs, architecture patterns | ✅ [H] | None |
| **Data quality / validation** | Expectations, monitoring | ⚠️ Brief | **HIGH GAP** |
| **Feedback loops** | Position bias, model-induced drift | ❌ Missing | **HIGH GAP** |
| Feature pipelines | Offline/online, freshness | ✅ [M] | None |
| Model monitoring | Drift detection, alerting | ✅ [H] | None |
| **Cold start** | New user/item strategies | ❌ Missing | **MEDIUM GAP** |

---

#### Breadth Area 8: Coding & SQL **[STRONG COMPETENCE]**

| Topic | Senior Bar | Current Agenda | Gap |
|-------|-----------|----------------|-----|
| Pandas/NumPy | Vectorized operations, efficient groupby | ✅ [C] | None |
| SQL window functions | ROW_NUMBER, LEAD/LAG, running totals | ✅ [H] | None |
| **SQL optimization** | Explain plans, indexing awareness | ⚠️ Implicit | Minor |
| DSA patterns | Arrays, trees, DP for AS | ✅ [H] | None |
| Clean code | Readable, testable, documented | ✅ [M] | None |

---

### Breadth Gap Summary for Senior Causal AS

| Gap | Severity | Recommendation |
|-----|----------|----------------|
| **Imbalanced learning** | HIGH | Add to Section 3.1 as **[H]** |
| **Model interpretability (SHAP, LIME)** | HIGH | Add to Section 3.1 or 3.5 as **[H]** |
| **Data quality / validation** | HIGH | Add to Section 6.2 as **[H]** |
| **Feedback loops / position bias** | HIGH | Add to Section 6.2 as **[H]** |
| Model calibration | MEDIUM | Upgrade 3.1.4 item to **[H]** |
| Feature selection methods | MEDIUM | Add to Section 3.4 as **[M]** |
| Transfer learning (non-LLM) | MEDIUM | Add to Section 3.3 as **[M]** |
| Cold start strategies | MEDIUM | Add to Section 6.3 as **[M]** |

---

### The "Breadth Round" in Senior Loops

Most big tech senior AS loops include **at least one general ML round** even for causal-focused roles:

```
Typical Senior Causal AS Loop (5-6 rounds):
├── Causal Inference Deep Dive (1-2 rounds)
├── ML/Coding (1 round) ← BREADTH TESTED HERE
├── System Design (1 round) ← BREADTH TESTED HERE
├── Case Study / Product (1 round)
└── Behavioral / Leadership (1 round)
```

**In the ML/Coding round, you might get:**
- "Build a classification model for fraud detection" (imbalanced learning)
- "This model's predictions are being used for ranking. What could go wrong?" (feedback loops)
- "Explain this model's decision to a regulator" (interpretability)

**In System Design round, you might get:**
- "Design a demand forecasting system" (end-to-end ML, not just causal)
- "Design a recommendation system with causal component" (breadth + causal integration)

---

### What "Senior" Breadth Looks Like vs IC Level

| Competency | IC (L4-L5) | Senior (L5-L6) | Staff+ (L7+) |
|------------|-----------|----------------|--------------|
| ML implementation | Can implement | Can critique and improve | Can define best practices |
| Feature engineering | Knows techniques | Knows trade-offs | Knows organizational patterns |
| System design | Can follow patterns | Can design from scratch | Can define architecture |
| Code quality | Writes good code | Reviews others' code | Sets standards |
| Trade-off articulation | Knows they exist | Can quantify | Can prioritize across org |

**The senior bar:** You need to demonstrate **judgment**, not just knowledge.

---

### Dimension 3: Experimentation Fluency (25% of evaluation)

**This is where many causal candidates underperform**

| Competency | Bar | Test Question Example |
|------------|-----|----------------------|
| A/B test design | Must master | "Design experiment for search ranking change" |
| Power analysis | Must master | "How many users do we need for 10% MDE?" |
| Variance reduction | Strong | "How would you reduce variance by 30%?" |
| Multiple testing | Strong | "We have 20 metrics. How do you control FDR?" |
| Sequential testing | Moderate | "We want to peek early. What changes?" |
| Interference | Moderate | "Users talk to each other. Now what?" |

**Big tech experimentation platforms differ:**
- Amazon: Heavy on power analysis, CUPED
- Meta: Focus on network effects, ranking experiments
- Netflix: Emphasis on long-term effects, surrogate metrics
- Uber/Lyft: Switchback designs, geographic experiments

---

### Dimension 4: Communication & Product Sense (10%)

| Competency | Bar | Example |
|------------|-----|---------|
| Translate to stakeholders | Strong | Explain p-value to PM without jargon |
| Trade-off articulation | Strong | "Why can't we just run an A/B test here?" |
| Scope definition | Strong | Know when causal question is worth answering |
| Failure communication | Moderate | "Our experiment was inconclusive. Now what?" |

---

### Company-Specific Bars

#### Amazon (Economics/Supply Chain AS)

| Area | Emphasis | Notes |
|------|----------|-------|
| Causal | **Very High** | Expect IV, DiD, strong econometrics |
| Experimentation | **Very High** | CUPED, variance reduction |
| ML | High | XGBoost for demand forecasting |
| Leadership Principles | **Very High** | Behavioral questions = 50% of loop |

**Unique expectations:** Optimization, demand estimation, price elasticity

---

#### Meta (Core Data Science)

| Area | Emphasis | Notes |
|------|----------|-------|
| Causal | **Very High** | Network effects, interference |
| Experimentation | **Very High** | Ranking experiments |
| ML | High | Recommendations, embeddings |
| Product sense | High | Metric definition heavy |

**Unique expectations:** Large-scale experiments, ecosystem effects

---

#### Netflix

| Area | Emphasis | Notes |
|------|----------|-------|
| Causal | High | Long-term effects |
| Experimentation | **Very High** | Surrogate metrics, generalization |
| ML | Moderate | Recommendations focus |
| Research depth | **Very High** | Publishes, wants thinkers |

**Unique expectations:** Scientific rigor, paper-quality thinking

---

#### Uber/Lyft (Marketplace)

| Area | Emphasis | Notes |
|------|----------|-------|
| Causal | High | Marketplace experiments |
| Experimentation | **Very High** | Switchback, interference |
| ML | High | Pricing, ETA prediction |
| System design | High | Real-time systems |

**Unique expectations:** Two-sided markets, geographic experiments

---

## Recommended Additions to Agenda

### Priority 1: Must Add (Critical)

```markdown
## 1.2.6 Propensity Score Methods **[C]**
- [ ] Propensity Score Matching **[C]**
  - Nearest neighbor (with/without replacement) **[C]**
  - Caliper matching **[H]**
  - Balance diagnostics (SMD, love plots) **[C]**
- [ ] Inverse Probability Weighting **[C]**
  - IPW estimator derivation **[C]**
  - Normalized vs unnormalized weights **[H]**
  - Weight trimming/truncation **[H]**
- [ ] Doubly Robust Estimation **[H]**
  - AIPW concept and advantages **[H]**
  - When it fails (both models wrong) **[M]**
- [ ] Diagnostics & Violations **[C]**
  - Overlap/positivity assessment **[C]**
  - Covariate balance verification **[C]**
```

### Priority 2: Must Add (High)

```markdown
## 1.2.1 (Expand) Randomized Experiments
- [ ] Variance Reduction **[H]**
  - CUPED (pre-experiment covariates) **[C]**
  - Stratified randomization **[H]**
  - Regression adjustment (Lin estimator) **[H]**
- [ ] Sequential & Adaptive **[M]**
  - Group sequential methods **[M]**
  - When to use vs fixed-horizon **[H]**
- [ ] Interference Handling **[M]**
  - Cluster randomization **[H]**
  - Switchback designs **[M]**
  - Two-sided markets considerations **[M]**
```

```markdown
## 1.7 Adaptive Experimentation **[H]** (NEW)
- [ ] Multi-Armed Bandits **[H]**
  - Thompson Sampling **[H]**
  - UCB algorithms **[M]**
  - When bandits vs A/B **[C]**
- [ ] Contextual Bandits **[H]**
  - Policy learning connection **[H]**
  - Off-policy evaluation (IPW, DR) **[M]**
```

### Priority 3: Should Add (Medium)

```markdown
## 1.2.7 Matching Methods **[H]** (NEW)
- [ ] Exact and Coarsened Exact Matching **[H]**
- [ ] Nearest Neighbor Matching **[H]**
- [ ] Matching vs Weighting trade-offs **[H]**
```

### Priority 4: Upgrade Existing

```markdown
## 1.5.1 Sensitivity Analysis **[H]** ← Upgrade from [M]
- [ ] E-values **[M]** ← Upgrade from [L]
- [ ] Benchmarking **[M]** (NEW)
```

---

## Summary: The Complete Causal AS Bar

| Dimension | Current Agenda | Gap | Post-Fix Target |
|-----------|---------------|-----|-----------------|
| Causal Foundations | **85%** | Minor | 95% |
| Classic Methods | **70%** | Propensity scores, matching | 95% |
| Modern Causal ML | **90%** | Minor | 95% |
| Experimentation | **50%** | CUPED, bandits, interference | 85% |
| ML Breadth | **80%** | Imbalanced learning, interpretability | 90% |
| Communication | **85%** | Minor | 90% |

**After fixing gaps, this agenda will be comprehensive for Causal AS roles at big tech.**
