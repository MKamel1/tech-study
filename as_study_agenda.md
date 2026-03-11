# Senior Applied Scientist Study Agenda

---
## Document Outline

### Execution Plan
- Constraints (20h/week, 28 weeks)
- 6-Phase Schedule (Complete Coverage)
- Portfolio Projects (2)
- Weekly Schedule Template

### 1. Causal Inference (Primary Spike)
- 1.1 Foundations [C]
- 1.2 Classic Methods [C] + 1.2.6 Propensity Score Methods [C]
- 1.3 Modern Causal ML [H]
- 1.4 Libraries & Tools [H] + Deliverables
- 1.5 Advanced Topics [H] (upgraded - sensitivity analysis)
- 1.6 Applications [H]

### 2. Probability, Statistics & Mathematical Foundations
- 2.1 Probability Basics [H] + MLE vs MAP
- 2.2 Distributions [H] + Delta Method
- 2.3 Statistical Inference [H] + Regression Diagnostics
- 2.4 Advanced Statistical Theory [M]
- 2.5 Linear Algebra for ML [H] *(NEW)*
- 2.6 Calculus & Optimization [H] *(NEW)*
- 2.7 Information Theory [M] *(NEW)*
- 2.8 Numerical Computation [L] *(NEW)*
- Section 2 Deliverables
- *Detailed plan: [02_probability_statistics_foundations.md](probability_statistics_linearAlgebra_foundations/02_probability_statistics_foundations.md)*

### 3. ML/DL Fundamentals
- 3.1 Supervised Learning [C] + 3.1.5 SVM
- 3.2 Unsupervised Learning [M] + 3.2.3 EM/GMM
- 3.3 Deep Learning Basics [H] + Backprop Derivation
- 3.4 Feature Engineering [H]
- 3.5 Model Selection & Tuning [H]
- Section 3 Deliverables

### 4. Time Series & Forecasting
- 4.1 Fundamentals [C]
- 4.2 Classical Methods [H]
- 4.3 Modern Methods [H]
- 4.4 Practical Considerations [H]
- 4.5 Uncertainty Quantification [M]
- 4.6 AS-Critical Topics [H] *(NEW: cold start, intermittent demand, regressors, ensembles, monitoring, business framing)*

### 5. NLP, RAG & Agentic AI
- 5.1 NLP Fundamentals [M]
- 5.2 Transformers & LLMs [H] + Attention Derivation
- 5.3 RAG [H]
- 5.4 Agentic AI [M]
- 5.5 Causal + AI Intersection [M]

### 6. ML System Design
- 6.1 Design Patterns [H]
- 6.2 Key Components [M] + Drift Metrics
- 6.3 Case Studies [H]

### 7. Coding & Algorithms
- 7.1 Core Patterns for AS [H]
- 7.2 Python for ML [H]
- 7.3 Coding Best Practices [M]

### 8. Product Sense & Communication
- 8.1 Problem Formulation [C]
- 8.2 Stakeholder Communication [H]
- 8.3 Case Study Practice [H]

### 9. Breadth Topics
- 9.1 Markov Models [M] + HMM Algorithms
- 9.2 Reinforcement Learning [L]
- 9.3 Survival Analysis [M]

### 10. Optional Topics
- 10.1 LLM Fine-Tuning [OPTIONAL]
- 10.2 Large-Scale Processing [OPTIONAL]
- 10.3 Deep Framework Proficiency [OPTIONAL]

### Quick Reference
- [Priority Matrix](#quick-reference-priority-matrix)
- [Study Approach by Priority](#study-approach-by-priority)
Comprehensive study curriculum for Senior Applied Scientist roles targeting economics/supply chain positions. Causal inference is the primary spike with supporting skills in ML, forecasting, and AI. Importance levels determine time investment: **[C]** Critical (deep study), **[H]** High (strong competence), **[M]** Medium (moderate effort), **[L]** Low (basics for breadth).

---

## Execution Plan

### Constraints
| Constraint | Value |
|------------|-------|
| Weekly hours | 20h |
| Timeline | 7 months (28 weeks) |
| Total hours | ~560h |
| Primary spike | Causal Inference (~100h) |
| Secondary spike | GenAI/LLM/RAG (~40h) |
| Projects | 2 |
| Mock interviews | 7+ |
| DSA practice | Daily (30min = ~3h/week) |

### Learning Principles Applied
| Principle | Implementation |
|-----------|----------------|
| **Interleaving** | Mix topics within phases (not sequential blocks) |
| **Spaced repetition** | Built-in review weeks (1-7-30 schedule) |
| **Active recall** | Test yourself before re-reading |
| **Verbal practice** | Record explanations, mock interviews |
| **Progressive interviews** | Mocks every 4 weeks → weekly at end |

---

### 6-Phase Schedule (Complete Coverage)

> **Design principles**: Interleaving for retention, realistic hours per subsection (~4-6h each), 
> dedicated slots for DSA review, mock interviews, and breadth topics.

---

#### Phase 0: Finish Time Series (Weeks 1-2) | 40h total
> Complete what you started before moving forward.

| Week | Focus | Sections | Hours |
|------|-------|----------|-------|
| 1 | Time Series Modern + Practice | 4.3-4.4 | 20h |
| 2 | Time Series Uncertainty + Review | 4.5 + 4.1-4.2 review | 20h |

**Weekly breakdown:**
- Theory: 10h | Practice: 6h | DSA review: 3h | Mock/review prep: 1h

**Deliverable**: Time series forecasting notebook with evaluation

---

#### Phase 1: Stats + Math Foundations + ML (Weeks 3-7) | 100h total
> Build the base everything else needs. Include breadth topic: Survival Analysis.
> **Expanded Section 2**: Includes linear algebra, calculus/optimization, and information theory foundations (validated against Goodfellow DL Part I).

| Week | Morning (Theory) | Afternoon (Practice) | Sections | Hours |
|------|------------------|---------------------|----------|-------|
| 3 | Stats 2.1 (Probability, Bayes, MLE/MAP) | DSA + probability problems | 2.1 | 20h |
| 4 | Stats 2.2 (Distributions) + Linear Algebra 2.5 | ML 3.1.1-3.1.2 | 2.2, 2.5, 3.1 | 20h |
| 5 | Stats 2.3 (Inference) + Calculus 2.6 | ML 3.1.3-3.1.4 (Trees, Eval) | 2.3, 2.6, 3.1 | 20h |
| 6 | Stats 2.4 + Info Theory 2.7 + Numerical 2.8 | ML 3.1.5-3.1.6 + **Survival Analysis 9.3** | 2.4, 2.7, 2.8, 3.1, 9.3 | 20h |
| 7 | SVD/Optimization deep dive + **Review** | Mock Interview #1 (Stats/ML) | 2.5-2.6 review, all | 20h |

**Weekly breakdown:**
- Theory: 8h | Practice/Code: 6h | DSA review: 4h | Breadth/Review: 2h

**Deliverables**:
- [ ] Derive MLE vs MAP on whiteboard
- [ ] Permutation test from scratch
- [ ] PCA from scratch using eigendecomposition (NumPy)
- [ ] Gradient descent for logistic regression from scratch
- [ ] XGBoost tuning notebook
- [ ] Kaplan-Meier survival curve implementation

**Mock Interview #1**: Week 7 - Stats + Math foundations + ML fundamentals

---

#### Phase 2: Causal Inference Deep Dive (Weeks 8-13) | 120h total
> Your primary spike. Go deep here.

| Week | Morning (Theory) | Afternoon (Practice) | Sections | Hours |
|------|------------------|---------------------|----------|-------|
| 8 | Causal 1.1 (Foundations) | DoWhy intro | 1.1 | 20h |
| 9 | Causal 1.2.1-1.2.3 (A/B, DiD, RDD) | Coding practice | 1.2.1-1.2.3 | 20h |
| 10 | Causal 1.2.4-1.2.5 (IV, Synthetic) | DoWhy notebooks | 1.2.4-1.2.5 | 20h |
| 11 | Causal 1.2.6-1.2.7 (PSM, Bandits) | EconML notebooks | 1.2.6-1.2.7 | 20h |
| 12 | Causal 1.3 (DML, Meta-learners) | Meta-learner comparison | 1.3 | 20h |
| 13 | Causal 1.4-1.5 (Libraries, Sensitivity) | Mock Interview #2 | 1.4-1.5 | 20h |

**Weekly breakdown:**
- Theory: 8h | Practice/Code: 6h | DSA review: 4h | Mock prep: 2h

**Deliverables**:
- [ ] Derive IV, DiD, RDD estimators on whiteboard
- [ ] DoWhy notebook with 3 identification strategies
- [ ] EconML meta-learner comparison
- [ ] Thompson Sampling implementation

**Mock Interview #2**: Week 13 - Causal deep dive

---

#### Phase 3: DL + ML Depth + Project 1 (Weeks 14-18) | 100h total
> Deep learning, remaining ML, causal applications, and first portfolio project.

| Week | Morning (Theory) | Afternoon (Practice) | Sections | Hours |
|------|------------------|---------------------|----------|-------|
| 14 | ML 3.3.1-3.3.2 (NN fundamentals) | Backprop from scratch | 3.3.1-3.3.2 | 20h |
| 15 | ML 3.3.3-3.3.4 (Architectures, Transfer) | PROJECT 1 start | 3.3.3-3.3.4 | 20h |
| 16 | ML 3.2 (Unsupervised) + 3.4 (Features) | PROJECT 1 continue | 3.2, 3.4 | 20h |
| 17 | ML 3.5 (Tuning, Interpretability) | PROJECT 1 continue | 3.5 | 20h |
| 18 | Causal 1.6 (Applications) + **Review** | PROJECT 1 finish + Mock #3 | 1.6 | 20h |

**Weekly breakdown:**
- Theory: 6h | Project: 8h | DSA review: 4h | Mock prep: 2h

**Deliverables**:
- [ ] Backprop derivation on whiteboard
- [ ] SHAP/LIME notebook
- [ ] PROJECT 1 complete on GitHub
- [ ] 5-minute interview pitch recorded

**Mock Interview #3**: Week 18 - Full causal + ML round

---

#### Phase 4: NLP/GenAI + Project 2 (Weeks 19-23) | 100h total
> Secondary spike: Complete NLP coverage including Agentic AI.

| Week | Morning (Theory) | Afternoon (Practice) | Sections | Hours |
|------|------------------|---------------------|----------|-------|
| 19 | NLP 5.1 (Fundamentals) + 5.2 (Transformers) | Attention block impl | 5.1, 5.2 | 20h |
| 20 | NLP 5.3 (RAG) | RAG prototype | 5.3 | 20h |
| 21 | NLP 5.4-5.5 (Agentic AI, Causal+AI) | PROJECT 2 start | 5.4, 5.5 | 20h |
| 22 | **Breadth: 9.1 (HMM) + 9.2 (RL basics)** | PROJECT 2 continue | 9.1, 9.2 | 20h |
| 23 | Coding 7.3 (Best Practices) + **Review** | PROJECT 2 finish + Mock #4 | 7.3 | 20h |

**Weekly breakdown:**
- Theory: 6h | Project: 8h | DSA review: 4h | Mock prep: 2h

**Deliverables**:
- [ ] Mini-transformer attention block
- [ ] RAG application
- [ ] Agentic workflow diagram
- [ ] PROJECT 2 complete on GitHub

**Mock Interview #4**: Week 23 - GenAI + system thinking

---

#### Phase 5: System Design + Interview Sprint (Weeks 24-28) | 100h total
> Polish everything. Weekly mocks. Heavy DSA.

| Week | Focus | Sections | Hours |
|------|-------|----------|-------|
| 24 | System Design 6.1-6.2 (Patterns, Components) | 6.1-6.2 | 20h |
| 25 | System Design 6.3 (Case Studies) | 6.3 | 20h |
| 26 | Coding 7.1-7.2 (Patterns, Python) intensive | 7.1-7.2 | 20h |
| 27 | Product Sense 8.1-8.3 | 8.1-8.3 | 20h |
| 28 | **Mock Interview Marathon** | Full review | 20h |

**Weekly breakdown:**
- Theory/Review: 6h | Practice problems: 8h | DSA intensive: 4h | Mock interview: 2h

**Mock Interviews**: Weekly from Week 24-28 (5 total)

---

### Hour Summary by Section

| Section | Subsections | Hours | Weeks |
|---------|-------------|-------|-------|
| 1. Causal Inference | 1.1-1.6 | ~100h | 8-13, 18 |
| 2. Stats & Math Foundations | 2.1-2.8 | ~55-60h | 3-7 |
| 3. ML/DL | 3.1-3.5 | ~80h | 4-6, 14-17 |
| 4. Time Series | 4.1-4.5 | ~40h | 1-2 |
| 5. NLP/GenAI | 5.1-5.5 | ~40h | 19-21 |
| 6. System Design | 6.1-6.3 | ~30h | 24-25 |
| 7. Coding | 7.1-7.3 | ~30h | 23, 26 |
| 8. Product Sense | 8.1-8.3 | ~20h | 27 |
| 9. Breadth | 9.1-9.3 | ~20h | 6, 22 |
| Projects | 2 projects | ~80h | 15-18, 21-23 |
| Mocks/Review | 7 mocks | ~40h | 7,13,18,23,24-28 |
| DSA (parallel) | Daily | ~80h | All weeks |
| **TOTAL** | | **~575h** | **28 weeks** |

> [!NOTE]
> Section 2 expanded from ~40h to ~55-60h after adding linear algebra, calculus, information theory, and numerical computation foundations. The additional ~15-20h is absorbed by interleaving math with ML in Phase 1 (weeks 4-7) rather than adding extra weeks. See [detailed Section 2 plan](probability_statistics_linearAlgebra_foundations/02_probability_statistics_foundations.md).



### Weekly Time Allocation Template (20h/week)

| Activity | Hours | Notes |
|----------|-------|-------|
| Core Theory | 6-8h | Reading, derivations, videos |
| Hands-on Practice | 6-8h | Code, notebooks, projects |
| DSA Review | 3-4h | Daily 30min + weekend problems |
| Mock/Review | 2h | Weekly review session or mock |
| Breadth/Buffer | 2h | Catch up, breadth topics |

### Daily Schedule Example
| Day | Activity | Hours |
|-----|----------|-------|
| Mon | Theory (morning) + DSA (30min) | 2.5h |
| Tue | Theory (morning) + DSA (30min) | 2.5h |
| Wed | Practice/Code + DSA (30min) | 3h |
| Thu | Practice/Code + DSA (30min) | 3h |
| Fri | Light review or rest | 1h |
| Sat | Deep work (project + theory) | 5h |
| Sun | Review + Mock prep + DSA | 3h |

### Portfolio Projects

#### Project 1: Causal Uplift Modeling (Weeks 12-15, ~60h)
> Demonstrate causal ML depth with real data.

| Component | Hours |
|-----------|-------|
| Dataset prep (Criteo/Hillstrom) | 8h |
| CATE estimation (4 meta-learners) | 20h |
| LLM subgroup discovery | 10h |
| Evaluation + GitHub repo | 14h |
| Interview story | 8h |

**Deliverables:**
- [ ] GitHub repo with CATE comparison
- [ ] 5-min interview pitch

#### Project 2: Choose at Week 16 (Weeks 17-20, ~50h)
> Select based on target companies.

**Option A: AV Safety Counterfactuals** (PhD leverage)
- Counterfactual trajectory analysis + DoWhy
- RAG for compliance checking
- Best for: Cruise, Zoox, Waymo

**Option B: CausalRAG** (GenAI depth)
- Causal graph + retrieval
- Best for: Google, Amazon, Meta

---

### Weekly Schedule Template (15-20h)

| Day | Hours | Focus |
|-----|-------|-------|
| Mon | 2-3h | Core theory (reading, derivations) |
| Tue | 2-3h | Core theory continued |
| Wed | 2-3h | Hands-on practice (code, notebooks) |
| Thu | 2-3h | Hands-on practice continued |
| Sat | 4-5h | Deep work block (projects) |
| Sun | 3-4h | Review + coding practice |

### Spaced Repetition Schedule (1-7-30)
After each major topic:
| When | Activity | Time |
|------|----------|------|
| Day 1 | Learn + implement | 2h |
| Day 2 | Teach it (record explanation) | 20min |
| Day 7 | Recall test (no notes) | 15min |
| Day 30 | Interview-style Q&A | 15min |

### Mock Interview Schedule
| Week | Type | Focus |
|------|------|-------|
| 12 | Causal deep dive | 1.1-1.5 methods |
| 16 | Causal + ML | Full technical |
| 20 | GenAI + System | Breadth test |
| 21-24 | Weekly full loops | All dimensions |

### Parallel Activities (Throughout)
| Activity | Frequency | When |
|----------|-----------|------|
| Coding practice | Daily | 30min AM warm-up |
| GitHub updates | Weekly | Push notebooks |
| LinkedIn/blog post | Biweekly | Document learnings |
| Peer discussion | Weekly | Study partner sync |

---

## Importance Legend & Time Strategy
| Label | Meaning | Interview Weight | Study Strategy |
|-------|---------|------------------|----------------|
| **[C]** | Critical - Must master | Asked in 80%+ of interviews | Deep study, hands-on projects |
| **[H]** | High - Strong competence | Asked in 50-80% of interviews | Solid understanding, practice |
| **[M]** | Medium - Should know well | Asked in 20-50% of interviews | Moderate effort, key concepts |
| **[L]** | Low - Familiarity sufficient | Rarely tested directly | Basics only, high-level discussion |

---

## 1. Causal Inference (Primary Spike)

> Your differentiation. Must be undeniably strong.

### 1.1 Foundations **[C]**
- [ ] 1.1.1 Potential Outcomes Framework (Rubin) **[C]**
  - Counterfactuals and treatment effects (ATE, ATT, CATE) **[C]**
  - SUTVA assumptions **[H]**
  - Assignment mechanisms **[H]**
- [ ] 1.1.2 Structural Causal Models (Pearl) **[C]**
  - Directed Acyclic Graphs (DAGs) **[C]**
  - d-separation and conditional independence **[C]**
  - do-calculus basics **[M]**
- [ ] 1.1.3 Identification Strategies **[C]**
  - Selection on observables vs unobservables **[C]**
  - Backdoor criterion **[C]**
  - Front-door criterion **[M]**

### 1.2 Classic Methods **[C]**
- [ ] 1.2.1 Randomized Experiments **[C]**
  - A/B testing design and analysis **[C]**
  - Power analysis and sample size **[C]**
  - Multiple testing corrections **[H]**
  - Variance reduction (CUPED, stratification) **[H]**
  - Regression adjustment (Lin estimator) **[H]**
  - Interference handling (switchback, cluster randomization) **[H]**
  - Post-stratification **[M]**
  - Sequential testing (group sequential, alpha spending) **[M]**
  - When to use sequential vs fixed-horizon **[H]**
- [ ] 1.2.2 Difference-in-Differences (DiD) **[C]**
  - Parallel trends assumption **[C]**
  - Event studies **[H]**
  - Staggered DiD (Callaway-Sant'Anna, Sun-Abraham) **[M]**
- [ ] 1.2.3 Regression Discontinuity (RDD) **[H]**
  - Sharp vs fuzzy RD **[H]**
  - Bandwidth selection **[M]**
  - Validity checks **[H]**
- [ ] 1.2.4 Instrumental Variables (IV) **[C]**
  - Relevance and exclusion conditions **[C]**
  - 2SLS implementation **[H]**
  - Weak instruments problem **[H]**
- [ ] 1.2.5 Synthetic Control Method **[C]**
  - Donor pool selection **[H]**
  - Pre-treatment fit **[H]**
  - Inference (permutation tests) **[M]**
- [ ] 1.2.6 Propensity Score Methods **[C]**
  - Propensity Score Matching (PSM) **[C]**
  - Inverse Probability Weighting (IPW) **[C]**
  - Doubly Robust Estimation **[H]**
  - Overlap/positivity violations **[H]**
  - Covariate balance diagnostics **[H]**
  - Matching methods (exact, CEM, nearest neighbor) **[H]**
  - Matching vs weighting trade-offs **[H]**
- [ ] 1.2.7 Adaptive Experimentation & Bandits **[H]**
  - Multi-Armed Bandit problem **[H]**
  - Exploration vs exploitation tradeoff **[H]**
  - Thompson Sampling **[H]**
  - UCB (Upper Confidence Bound) **[M]**
  - Epsilon-Greedy **[M]**
  - Contextual Bandits **[M]**
  - When to use bandits vs A/B tests **[H]**
  - Off-Policy Evaluation **[M]**
  - IPW/DR for counterfactual estimates **[M]**

### 1.3 Modern Causal ML **[H]**
- [ ] 1.3.1 Double/Debiased Machine Learning (DML) **[H]**
  - Cross-fitting procedure **[H]**
  - Nuisance parameter estimation **[M]**
  - When to use vs traditional methods **[C]**
- [ ] 1.3.2 Causal Forests **[H]**
  - Heterogeneous treatment effects **[H]**
  - Honest estimation **[M]**
  - Variable importance for effect modifiers **[M]**
- [ ] 1.3.3 Meta-Learners **[H]**
  - S-learner, T-learner, X-learner **[H]**
  - DR-learner **[M]**
  - Trade-offs and when to use each **[C]**

### 1.4 Libraries & Tools **[H]**
- [ ] 1.4.1 DoWhy **[H]**
  - Causal graph specification **[H]**
  - Identification and estimation **[H]**
  - Refutation tests **[H]**
- [ ] 1.4.2 EconML **[H]**
  - DML implementation **[H]**
  - Causal forests **[H]**
  - Policy learning **[M]**
- [ ] 1.4.3 CausalML **[M]**
  - Uplift modeling **[H]**
  - Meta-learners **[M]**

#### Section 1 Deliverables
- [ ] Derive IV, DiD, RDD estimators on whiteboard
- [ ] DoWhy notebook with 3 identification strategies
- [ ] Implement S-learner, T-learner from scratch
- [ ] EconML comparison notebook (all 4 meta-learners on same dataset)

### 1.5 Advanced Topics **[H]**
- [ ] 1.5.1 Sensitivity Analysis **[H]**
  - Omitted variable bias bounds **[H]**
  - Rosenbaum bounds **[M]**
  - E-values **[H]**
  - Benchmarking to observed confounders **[M]**
- [ ] 1.5.2 Partial Identification **[L]**
  - Bounds without point identification **[L]**
  - Manski bounds **[L]**
- [ ] 1.5.3 Mediation Analysis **[M]**
  - Direct and indirect effects **[M]**
  - Causal mediation analysis **[M]**
- [ ] 1.5.4 Granger Causality **[L]** *(NEW)*
  - Temporal precedence test (NOT true causation) **[L]**
  - When useful: lead-lag relationships in time series **[L]**
  - Common interview gotcha: "Granger ≠ true causation" **[L]**

### 1.6 Applications **[H]**
- [ ] 1.6.1 Marketing & Uplift **[H]**
  - Uplift modeling for targeting **[H]**
  - Customer lifetime value attribution **[M]**
- [ ] 1.6.2 Economics-Based Prediction **[H]**
  - Policy evaluation **[H]**
  - Demand estimation **[H]**
- [ ] 1.6.3 Supply Chain Causal Problems **[H]**
  - Supplier effect estimation
  - Intervention analysis

---

## 2. Probability, Statistics & Mathematical Foundations

> Foundation for all quantitative work. Must be solid.
> **Detailed plan**: [02_probability_statistics_foundations.md](probability_statistics_linearAlgebra_foundations/02_probability_statistics_foundations.md) (~55-60h, calibrated against Goodfellow DL Part I)

### 2.1 Probability Basics **[H]**
- [ ] 2.1.1 Combinatorics **[M]**
  - Permutations and combinations **[M]**
  - Counting principles **[M]**
- [ ] 2.1.2 Probability Rules **[H]**
  - Conditional probability **[H]**
  - Independence **[H]**
  - Law of total probability **[M]**
- [ ] 2.1.3 Random Variables & Expectations **[H]**
  - PMF, PDF, CDF **[H]**
  - Expectation, variance, covariance **[C]**
- [ ] 2.1.4 Bayes' Theorem **[C]**
  - Intuition and formula **[C]**
  - Prior, likelihood, posterior **[H]**
  - Common interview applications **[C]**
  - Conjugate priors (Beta-Binomial example) **[M]**
- [ ] 2.1.5 MLE vs MAP **[H]**
  - Derive MLE vs MAP (when equal, why) **[H]**
  - MAP-regularization equivalence (Gaussian→L2, Laplace→L1) **[C]**
- [ ] 2.1.6 Probability Inequalities **[M]**
  - Markov, Chebyshev, Jensen's **[M-H]**

### 2.2 Distributions **[H]**
- [ ] 2.2.1 Discrete Distributions **[H]**
  - Bernoulli, Binomial, Poisson, Geometric, Multinomial **[H]**
  - When to use each **[C]**
- [ ] 2.2.2 Continuous Distributions **[H]**
  - Normal (Gaussian) **[C]**
  - Exponential, Uniform, Log-Normal, Beta **[M]**
  - Chi-squared, t-distribution, F-distribution **[H]**
  - Multivariate Normal **[H]**
- [ ] 2.2.3 Key Theorems **[H]**
  - Central Limit Theorem (intuition + application) **[C]**
  - Law of Large Numbers (weak vs strong) **[H]**
  - Delta method for variance **[M]**
- [ ] 2.2.4 Distribution Relationships **[M]**
  - Exponential family unification **[M]**
  - Key relationships diagram **[M]**

### 2.3 Statistical Inference **[H]**
- [ ] 2.3.1 Estimation **[H]**
  - Point estimates, confidence intervals **[C]**
  - Standard error, bootstrap methods **[H]**
- [ ] 2.3.2 Hypothesis Testing **[C]**
  - Null/alternative hypotheses, p-values **[C]**
  - Type I/II errors, power analysis **[C-H]**
  - Multiple testing corrections (Bonferroni, FDR) **[H]**
- [ ] 2.3.3 Common Tests **[H]**
  - t-tests, Chi-square, ANOVA **[H-M]**
  - Non-parametric tests, permutation tests **[M-H]**
- [ ] 2.3.4 Regression Foundations **[H]**
  - OLS and logistic regression **[C]**
  - GLMs (conceptual) **[M]**
- [ ] 2.3.5 Regression Diagnostics **[H]**
  - Heteroscedasticity (Breusch-Pagan, White test) **[H]**
  - Multicollinearity (VIF) **[H]**
  - Autocorrelation (Durbin-Watson) **[M]**
  - Influential points (Cook's distance) **[M]**

### 2.4 Advanced Statistical Theory **[M]**
- [ ] 2.4.1 Asymptotic Theory **[M]**
  - Consistency, asymptotic normality **[M]**
  - Key convergence concepts (in probability, in distribution) **[M]**
- [ ] 2.4.2 Estimation Theory **[L-M]**
  - Fisher Information, Cramer-Rao bound (conceptual) **[L-M]**
- [ ] 2.4.3 Bayesian Foundations **[M]**
  - MCMC intuition (Metropolis-Hastings, Gibbs) **[L]**
  - Credible vs confidence intervals **[M]**
- [ ] 2.4.4 Causal & Statistical Connections **[M]**
  - Simpson's Paradox, Berkson's Paradox **[H-M]**

### 2.5 Linear Algebra for ML **[H]** *(NEW)*
- [ ] 2.5.1 Vectors, matrices, norms, determinant, trace **[H]**
- [ ] 2.5.2 Eigendecomposition **[H]**
- [ ] 2.5.3 Singular Value Decomposition (SVD) **[H]**
- [ ] 2.5.4 Positive definite matrices **[M]**
- [ ] 2.5.5 Matrix calculus essentials **[M]**

### 2.6 Calculus & Optimization **[H]** *(NEW)*
- [ ] 2.6.1 Multivariate calculus (gradients, chain rule) **[H]**
- [ ] 2.6.2 Convexity **[H]**
- [ ] 2.6.3 Gradient descent, SGD, Newton's method **[H]**
- [ ] 2.6.4 Constrained optimization (Lagrange multipliers) **[M]**
- [ ] 2.6.5 Integration essentials **[M]**

### 2.7 Information Theory **[M]** *(NEW)*
- [ ] 2.7.1 Entropy **[H]**
- [ ] 2.7.2 Cross-entropy (why it's the classification loss) **[C]**
- [ ] 2.7.3 KL divergence **[H]**
- [ ] 2.7.4 Mutual information **[M]**

### 2.8 Numerical Computation **[L]** *(NEW)*
- [ ] 2.8.1 Overflow/underflow, log-sum-exp trick **[M]**
- [ ] 2.8.2 Conditioning and regularization **[M]**
- [ ] 2.8.3 Automatic differentiation (conceptual) **[M]**

#### Section 2 Deliverables
- [ ] Derive MLE for Normal/Bernoulli + MAP-regularization equivalence
- [ ] Implement permutation test from scratch
- [ ] Implement bootstrap CI from scratch
- [ ] Derive OLS normal equations on whiteboard
- [ ] PCA from scratch using eigendecomposition (NumPy)
- [ ] Gradient descent for logistic regression from scratch
- [ ] Derive cross-entropy loss from MLE
- [ ] Log-sum-exp stability demo

---

## 3. ML/DL Fundamentals

> Core competence for any AS role.

### 3.1 Supervised Learning **[C]**
- [ ] 3.1.1 Core Concepts **[C]**
  - Bias-variance tradeoff (must explain intuitively) **[C]**
  - Overfitting and underfitting **[C]**
  - Train/validation/test splits **[C]**
  - Cross-validation strategies **[H]**
- [ ] 3.1.2 Linear Models **[C]**
  - Linear/logistic regression **[C]**
  - Regularization (L1, L2, ElasticNet) **[C]**
  - Feature scaling importance **[H]**
- [ ] 3.1.3 Tree-Based Methods **[C]**
  - Decision trees (splitting criteria, pruning) **[H]**
  - Random Forests (bagging, feature importance) **[C]**
  - Gradient Boosting (XGBoost, LightGBM, CatBoost) **[C]**
  - Hyperparameter tuning strategies **[H]**
- [ ] 3.1.4 Model Evaluation **[C]**
  - Classification: precision, recall, F1, AUC-ROC, AUC-PR **[C]**
  - Regression: MSE, MAE, MAPE, R-squared **[C]**
  - Calibration and reliability diagrams **[H]**
  - Platt scaling, isotonic regression **[H]**
  - Confusion matrix interpretation **[C]**
- [ ] 3.1.5 Support Vector Machines **[M]**
  - Margin maximization intuition **[M]**
  - Kernel trick (map to high dimensions) **[M]**
  - Common kernels (linear, RBF, polynomial) **[M]**
  - C and gamma hyperparameters **[L]**
- [ ] 3.1.6 Imbalanced Learning **[H]**
  - Class imbalance impact on metrics **[H]**
  - SMOTE and variants **[H]**
  - Class weights and cost-sensitive learning **[H]**
  - Threshold tuning for precision-recall **[H]**
  - AUC-PR vs AUC-ROC for imbalanced data **[C]**

### 3.2 Unsupervised Learning **[M]**
- [ ] 3.2.1 Clustering **[M]**
  - K-means (initialization, elbow method) **[H]**
  - Hierarchical clustering **[M]**
  - DBSCAN **[M]**
- [ ] 3.2.2 Dimensionality Reduction **[M]**
  - PCA (intuition, eigendecomposition, SVD) **[H]**
  - ICA (independence vs correlation, when to use) **[M]**
  - t-SNE and UMAP (visualization) **[M]**
- [ ] 3.2.3 EM Algorithm & Mixture Models **[M]**
  - Latent variable motivation (soft clustering) **[M]**
  - E-step: compute responsibilities **[M]**
  - M-step: update parameters **[M]**
  - GMM for clustering **[M]**
  - Convergence properties (local optima) **[L]**

### 3.3 Deep Learning Basics **[H]**
- [ ] 3.3.1 Neural Network Fundamentals **[H]**
  - Forward and backward propagation **[H]**
  - Derive backprop from chain rule (computational graph) **[H]**
  - Vanishing/exploding gradients (causes & solutions) **[H]**
  - Activation functions (ReLU, sigmoid, softmax) **[H]**
  - Loss functions **[H]**
  - Optimizers (SGD, Adam, momentum intuition) **[M]**
- [ ] 3.3.2 Practical Considerations **[M]**
  - Batch normalization (what & WHY it works) **[M]**
  - Dropout (what & WHY it works as regularization) **[H]**
  - Weight initialization strategies **[M]**
  - Learning rate scheduling **[M]**
  - Early stopping **[H]**
- [ ] 3.3.3 Architectures Overview **[H]**
  - MLPs for tabular data **[H]**
  - CNNs for images (conceptual) **[M]**
  - RNNs/LSTMs for sequences (conceptual) **[M]**
  - Transformers (attention mechanism intuition) **[H]**
- [ ] 3.3.4 Transfer Learning **[M]**
  - When to freeze vs fine-tune layers **[M]**
  - Domain adaptation concepts **[L]**
  - Pretrained embeddings (non-LLM) **[M]**

### 3.4 Feature Engineering **[H]**
- [ ] 3.4.1 Numerical Features **[H]**
  - Scaling, normalization **[H]**
  - Binning, log transforms **[M]**
  - Handling outliers **[H]**
- [ ] 3.4.2 Categorical Features **[H]**
  - One-hot encoding **[H]**
  - Target encoding **[M]**
  - Embedding approaches **[M]**
- [ ] 3.4.3 Missing Data **[H]**
  - Imputation strategies **[H]**
  - Missingness patterns (MCAR, MAR, MNAR) **[M]**
- [ ] 3.4.4 Feature Selection **[M]**
  - Filter methods (correlation, mutual information) **[M]**
  - Wrapper methods (RFE) **[M]**
  - Embedded methods (L1, tree importance) **[H]**

### 3.5 Model Selection & Tuning **[H]**
- [ ] 3.5.1 Hyperparameter Optimization **[H]**
  - Grid search, random search **[H]**
  - Bayesian optimization (conceptual) **[M]**
- [ ] 3.5.2 Model Comparison **[H]**
  - Statistical tests for model comparison **[M]**
  - Ensemble methods **[H]**
- [ ] 3.5.3 Model Interpretability **[H]**
  - SHAP values (intuition and use) **[H]**
  - LIME (local interpretability) **[H]**
  - Feature importance methods **[H]**
  - Explaining predictions to stakeholders **[C]**

#### Section 3 Deliverables
- [ ] Implement 2-layer NN from scratch (NumPy only)
- [ ] Derive backpropagation on whiteboard
- [ ] Implement dropout and batch norm manually
- [ ] Build mini-transformer attention block

---

## 4. Time Series & Forecasting

> Critical for demand/supply chain roles.

### 4.1 Fundamentals **[C]**
- [ ] 4.1.1 Time Series Components **[C]**
  - Trend, seasonality, cycles, noise **[C]**
  - Additive vs multiplicative decomposition **[H]**
  - STL decomposition **[H]**
- [ ] 4.1.2 Stationarity **[C]**
  - Definition and importance **[C]**
  - ADF test, KPSS test **[H]**
  - Differencing for stationarity **[H]**

### 4.2 Classical Methods **[H]**
- [ ] 4.2.1 ARIMA Family **[H]**
  - AR, MA, ARMA, ARIMA components **[H]**
  - ACF/PACF interpretation **[H]**
  - Model selection (AIC, BIC) **[M]**
  - SARIMA for seasonality **[H]**
  - ARIMAX (external regressors) **[H]** *(NEW)*
  - Auto-ARIMA (pmdarima) **[M]** *(NEW)*
- [ ] 4.2.2 Exponential Smoothing **[M]**
  - Simple, Holt, Holt-Winters **[M]**
  - ETS framework **[M]**
- [ ] 4.2.3 Vector Autoregression (VAR) **[L]** *(NEW)*
  - Multivariate time series modeling **[L]**
  - Impulse response functions (conceptual) **[L]**
  - Use case: cross-series dependencies **[L]**

### 4.3 Modern Methods **[H]**
- [ ] 4.3.1 Prophet **[H]**
  - Additive model components **[H]**
  - Handling holidays and events **[M]**
  - Changepoint detection **[M]**
  - When to use vs alternatives **[C]**
- [ ] 4.3.2 ML-Based Forecasting **[H]**
  - Feature engineering for time series **[H]**
  - XGBoost/LightGBM for forecasting **[H]**
  - Lag features, rolling statistics **[H]**
  - Global vs Local models (cross-learning vs per-series) **[H]**
- [ ] 4.3.3 Neural Forecasters **[M]**
  - N-BEATS (conceptual) **[L]**
  - Temporal Fusion Transformer (conceptual) **[L]**

### 4.4 Practical Considerations **[H]**
- [ ] 4.4.1 Evaluation Metrics **[C]**
  - MAPE, SMAPE, RMSE, MAE **[C]**
  - MASE (scale-free, naive baseline) **[H]**
  - When each is appropriate **[C]**
  - Weighted metrics for hierarchies **[M]**
- [ ] 4.4.2 Cross-Validation for Time Series **[H]**
  - Walk-forward validation **[H]**
  - Expanding vs sliding window **[H]**
- [ ] 4.4.3 Multiple Seasonalities **[M]**
  - Daily + weekly + yearly **[M]**
  - TBATS, Prophet, Fourier terms **[M]**
- [ ] 4.4.4 Hierarchical Forecasting **[M]**
  - Top-down, bottom-up, middle-out **[M]**
  - Reconciliation methods **[L]**

### 4.5 Uncertainty Quantification **[M]**
- [ ] 4.5.1 Prediction Intervals **[M]**
  - Parametric intervals **[M]**
  - Bootstrapping **[M]**
  - Conformal prediction **[L]**

### 4.6 AS-Critical Topics **[H]**
> Essential for supply chain/demand forecasting AS interviews

- [ ] 4.6.1 Cold Start / New Product Forecasting **[H]**
  - Similar product matching **[H]**
  - Attribute-based models **[H]**
  - Warm-up period strategies **[M]**
- [ ] 4.6.2 Intermittent Demand **[H]**
  - Croston's method **[H]**
  - Syntetos-Boylan Approximation (SBA) **[M]**
  - When standard methods fail **[H]**
- [ ] 4.6.3 External Regressors **[H]**
  - Promotions, price, weather, events **[H]**
  - Known vs unknown future regressors **[H]**
  - ARIMAX and Prophet with regressors **[M]**
- [ ] 4.6.4 Forecast Ensembles & Combinations **[H]**
  - Simple and weighted averaging **[H]**
  - Stacking methods **[M]**
  - M-competition insights **[M]**
- [ ] 4.6.5 Anomaly Detection in Time Series **[M]**
  - Statistical and model-based methods **[M]**
  - Handling strategies (remove, impute, flag) **[M]**
- [ ] 4.6.6 Missing Data in Time Series **[M]**
  - Imputation strategies **[M]**
  - Random vs systematic missingness **[M]**
- [ ] 4.6.7 Model Monitoring & Retraining **[H]**
  - Concept drift and data drift **[H]**
  - Retraining triggers and strategies **[H]**
  - Rolling vs expanding window **[M]**
- [ ] 4.6.8 Business Framing for Forecasting **[C]**
  - Problem formulation **[C]**
  - Horizon and granularity selection **[C]**
  - Asymmetric cost trade-offs **[H]**
  - Stakeholder communication **[H]**

---

## 5. NLP, RAG & Agentic AI

> Differentiator and fallback option.

### 5.1 NLP Fundamentals **[M]**
- [ ] 5.1.1 Text Preprocessing **[M]**
  - Tokenization, stemming, lemmatization **[M]**
  - Stop words, n-grams **[L]**
- [ ] 5.1.2 Text Representations **[M]**
  - Bag of Words, TF-IDF **[M]**
  - Word embeddings (Word2Vec, GloVe) **[M]**
  - Sentence embeddings **[H]**
- [ ] 5.1.3 Common Tasks **[M]**
  - Text classification **[M]**
  - Named Entity Recognition (NER) **[M]**
  - Sentiment analysis **[M]**
  - Document similarity **[M]**

### 5.2 Transformers & LLMs **[H]**
- [ ] 5.2.1 Transformer Architecture **[H]**
  - Self-attention mechanism (intuition + derivation) **[H]**
  - Scaled dot-product attention formula **[H]**
  - Multi-head attention (WHY multiple heads) **[H]**
  - Positional encoding (WHY needed) **[M]**
  - Encoder vs decoder **[H]**
  - Attention complexity O(n²) **[M]**
- [ ] 5.2.2 LLM Landscape **[H]**
  - GPT, BERT, T5 differences **[H]**
  - Open vs closed models **[M]**
  - Model selection criteria **[H]**
- [ ] 5.2.3 Using LLMs **[H]**
  - Prompt engineering basics **[H]**
  - Few-shot learning **[H]**
  - Temperature and sampling **[M]**

### 5.3 RAG (Retrieval-Augmented Generation) **[H]**
- [ ] 5.3.1 RAG Architecture **[H]**
  - Embedding models **[H]**
  - Vector databases (conceptual) **[M]**
  - Retrieval strategies **[H]**
  - Generation with context **[H]**
- [ ] 5.3.2 Practical Implementation **[M]**
  - Chunking strategies **[M]**
  - Metadata filtering **[M]**
  - Reranking **[M]**
- [ ] 5.3.3 RAG Evaluation **[H]**
  - Retrieval metrics (recall, MRR) **[M]**
  - Generation quality **[H]**
  - Hallucination detection **[H]**

### 5.4 Agentic AI **[M]**
- [ ] 5.4.1 Agent Patterns **[M]**
  - ReAct pattern **[M]**
  - Tool use and function calling **[H]**
  - Planning and reasoning **[M]**
- [ ] 5.4.2 Agent Frameworks **[M]**
  - LangChain/LlamaIndex basics **[M]**
  - Multi-agent systems (conceptual) **[L]**
  - MCP (Model Context Protocol) basics **[M]**
- [ ] 5.4.3 Agent Challenges **[M]**
  - Reliability and error handling **[M]**
  - Human-in-the-loop **[H]**
  - Evaluation strategies **[M]**

### 5.5 Causal + AI Intersection **[M]**
- [ ] 5.5.1 LLM Evaluation with Causal Lens **[M]**
  - Confounding in LLM benchmarks **[L]**
  - Causal approaches to evaluation **[M]**
- [ ] 5.5.2 LLM-Assisted Causal Discovery **[M]**
  - LLMs for hypothesis generation **[M]**
  - Hybrid human-AI causal analysis **[M]**

---

## 6. ML System Design

> Required for senior roles.

### 6.1 Design Patterns **[H]**
- [ ] 6.1.1 Batch Prediction Systems **[H]**
  - Offline scoring pipelines **[H]**
  - Feature computation **[M]**
  - Model refresh strategies **[M]**
- [ ] 6.1.2 Real-Time Prediction Systems **[H]**
  - Online serving architecture **[H]**
  - Latency considerations **[H]**
  - Caching strategies **[M]**
- [ ] 6.1.3 Training Pipelines **[M]**
  - Data ingestion **[M]**
  - Feature engineering **[H]**
  - Model training and validation **[H]**
  - Artifact management **[L]**

### 6.2 Key Components **[M]**
- [ ] 6.2.1 Feature Stores **[M]**
  - Online vs offline features **[M]**
  - Feature freshness **[L]**
  - Point-in-time correctness **[M]**
- [ ] 6.2.2 Model Serving **[M]**
  - REST APIs vs batch **[M]**
  - Containerization (Docker basics) **[M]**
  - Scaling considerations **[L]**
- [ ] 6.2.3 Monitoring & Observability **[H]**
  - Model performance monitoring **[H]**
  - Data drift detection **[H]**
  - Drift metrics (PSI, KL divergence, KS test) **[M]**
  - Alerting strategies **[M]**
- [ ] 6.2.4 Data Quality & Validation **[H]**
  - Data expectations (Great Expectations, Deequ) **[M]**
  - Schema validation **[M]**
  - Data freshness and completeness checks **[H]**
  - Upstream data change detection **[H]**
- [ ] 6.2.5 Feedback Loops & Bias **[H]**
  - Position bias in rankings **[H]**
  - Model-induced distribution shift **[H]**
  - Feedback loop identification **[H]**
  - Debiasing strategies (IPW, position models) **[M]**

### 6.3 Case Studies **[H]**
- [ ] 6.3.1 Recommendation System **[H]**
  - Candidate generation + ranking **[H]**
  - Feature engineering for recommendations **[M]**
- [ ] 6.3.2 Demand Forecasting System **[C]**
  - Hierarchical predictions **[H]**
  - Serving forecasts to downstream systems **[M]**
- [ ] 6.3.3 Fraud/Anomaly Detection **[M]**
  - Class imbalance handling **[H]**
  - Real-time scoring requirements **[M]**
- [ ] 6.3.4 Cold Start Strategies **[M]**
  - New user/item handling **[M]**
  - Content-based fallbacks **[M]**
  - Exploration strategies **[M]**

---

## 7. Coding & Algorithms

> Maintenance mode - you've already covered DSA.

### 7.1 Core Patterns for AS **[H]**
- [ ] 7.1.1 Arrays & Hashing **[H]**
  - Two pointers, sliding window **[H]**
  - Hash maps for lookups **[H]**
- [ ] 7.1.2 Trees **[H]**
  - Binary tree traversals **[H]**
  - BST operations **[M]**
  - Common tree problems **[H]**
- [ ] 7.1.3 Dynamic Programming **[H]**
  - 1D and 2D DP patterns **[H]**
  - Memoization vs tabulation **[H]**
  - Common problems (knapsack, LIS, paths) **[H]**
- [ ] 7.1.4 Graphs **[M]**
  - BFS/DFS **[M]**
  - Basic shortest path **[L]**

### 7.2 Python for ML **[H]**
- [ ] 7.2.1 NumPy/Pandas Proficiency **[C]**
  - Vectorized operations **[H]**
  - GroupBy and aggregations **[C]**
  - Merging and joins **[H]**
- [ ] 7.2.2 Scikit-learn Fluency **[H]**
  - Pipeline construction **[H]**
  - Custom transformers **[M]**
  - Model persistence **[M]**
- [ ] 7.2.3 SQL Proficiency **[H]**
  - Window functions **[H]**
  - CTEs **[M]**
  - Complex joins and aggregations **[H]**

### 7.3 Coding Best Practices **[M]**
- [ ] 7.3.1 Clean Code **[M]**
  - Readable, modular code **[M]**
  - Error handling **[M]**
  - Type hints **[L]**
- [ ] 7.3.2 Testing Basics **[M]**
  - Unit tests **[M]**
  - Data validation **[M]**

---

## 8. Product Sense & Communication

> What makes you senior.

### 8.1 Problem Formulation **[C]**
- [ ] 8.1.1 Business to Technical Translation **[C]**
  - Identifying the right metric **[C]**
  - Defining success criteria **[C]**
  - Scoping problems appropriately **[H]**
- [ ] 8.1.2 When NOT to Use ML **[C]**
  - Rule-based alternatives **[H]**
  - Cost-benefit analysis **[C]**
  - Technical debt considerations **[M]**

### 8.2 Stakeholder Communication **[H]**
- [ ] 8.2.1 Explaining Models **[H]**
  - Non-technical explanations **[H]**
  - Uncertainty communication **[H]**
  - Limitations and caveats **[H]**
- [ ] 8.2.2 Trade-off Discussions **[H]**
  - Accuracy vs interpretability **[H]**
  - Speed vs accuracy **[M]**
  - Build vs buy **[M]**

### 8.3 Case Study Practice **[H]**
- [ ] 8.3.1 Metric Definition Cases **[H]**
  - "How would you measure success for X?" **[H]**
- [ ] 8.3.2 Ambiguous Problem Cases **[H]**
  - "We have problem X, what would you do?" **[H]**
- [ ] 8.3.3 Trade-off Cases **[H]**
  - "Our model has issue Y, how do you handle it?" **[H]**

---

## 9. Breadth Topics (Low Priority)

> Basics only for high-level discussion. Spend minimal time.

### 9.1 Markov Models **[M]**
- [ ] 9.1.1 Markov Chains **[M]**
  - State transition basics **[M]**
  - Transition matrix **[M]**
  - Stationary distributions (conceptual) **[L]**
  - When they're used **[M]**
- [ ] 9.1.2 Hidden Markov Models **[M]**
  - Hidden states vs observations **[M]**
  - Three canonical problems (evaluation, decoding, learning) **[M]**
  - Viterbi algorithm (dynamic programming decoding) **[M]**
  - Forward-backward algorithm (conceptual) **[L]**
  - Baum-Welch = EM for HMMs (conceptual) **[L]**

### 9.2 Reinforcement Learning **[L]**
- [ ] 9.2.1 Core Concepts Only **[L]**
  - Agent, environment, reward **[L]**
  - Exploration vs exploitation **[L]**
  - Policy vs value functions (conceptual) **[L]**
- [ ] 9.2.2 When RL is Used **[L]**
  - Recommendation systems **[L]**
  - Robotics, gaming **[L]**
  - When NOT to use RL **[L]**

### 9.3 Survival Analysis **[M]**
- [ ] 9.3.1 Fundamentals **[M]**
  - Time-to-event modeling **[M]**
  - Censoring (right, left, interval) **[M]**
  - Survival function S(t), Hazard function h(t) **[M]**
- [ ] 9.3.2 Methods **[M]**
  - Kaplan-Meier estimator **[M]**
  - Log-rank test (comparing groups) **[L]**
  - Cox proportional hazards (conceptual) **[M]**

---

## 10. Optional Topics

> Complete main sections first. Visit these only after finishing everything else.

### 10.1 LLM Fine-Tuning **[OPTIONAL]**
- [ ] 10.1.1 Fine-Tuning Approaches **[OPTIONAL]**
  - Full fine-tuning vs parameter-efficient (LoRA, QLoRA) **[OPTIONAL]**
  - When to fine-tune vs prompt engineering **[OPTIONAL]**
- [ ] 10.1.2 Practical Fine-Tuning **[OPTIONAL]**
  - Dataset preparation **[OPTIONAL]**
  - Training considerations **[OPTIONAL]**
  - Evaluation strategies **[OPTIONAL]**
- [ ] 10.1.3 Open Models **[OPTIONAL]**
  - Llama family overview **[OPTIONAL]**
  - Mistral, Phi overview **[OPTIONAL]**

### 10.2 Large-Scale Processing **[OPTIONAL]**
- [ ] 10.2.1 PySpark Basics **[OPTIONAL]**
  - DataFrame operations **[OPTIONAL]**
  - Spark ML overview **[OPTIONAL]**
  - When to use Spark vs Pandas **[OPTIONAL]**
- [ ] 10.2.2 Distributed Training Concepts **[OPTIONAL]**
  - Data parallelism **[OPTIONAL]**
  - Model parallelism (conceptual) **[OPTIONAL]**

### 10.3 Deep Framework Proficiency **[OPTIONAL]**
- [ ] 10.3.1 PyTorch **[OPTIONAL]**
  - Tensor operations **[OPTIONAL]**
  - Building simple models **[OPTIONAL]**
  - Training loops **[OPTIONAL]**
- [ ] 10.3.2 TensorFlow/Keras **[OPTIONAL]**
  - High-level API usage **[OPTIONAL]**
  - Model building basics **[OPTIONAL]**

---

## Quick Reference: Priority Matrix

| Area | Critical [C] | High [H] | Medium [M] | Low [L] |
|------|-------------|----------|------------|---------|
| **Causal Inference** | Foundations, Classic Methods | Causal ML, Libraries, Applications | Advanced Topics | - |
| **Prob, Stats & Math** | Cross-entropy | Probability, Distributions, Inference, LinAlg, Optimization | Advanced Theory, Info Theory | Numerical Computation |
| **ML/DL** | Supervised Learning | DL Basics, Feature Eng, Tuning | Unsupervised | - |
| **Time Series** | Fundamentals | Classical, Modern, Practical | Neural, Hierarchical, UQ | - |
| **NLP/RAG/Agentic** | - | Transformers, RAG | NLP Basics, Agentic, Causal+AI | - |
| **System Design** | - | Patterns, Case Studies | Components | - |
| **Coding** | - | Core Patterns, Python/SQL | Graphs, Best Practices | - |
| **Product Sense** | Problem Formulation | Communication, Cases | - | - |
| **Breadth** | - | - | - | Markov, RL |
| **Optional** | - | - | - | Fine-tuning, Spark, Frameworks |

---

## Study Approach by Priority

| Priority | Depth | Approach |
|----------|-------|----------|
| **[C] Critical** | Mastery | Deep study, hands-on projects, practice explaining, must be able to teach others |
| **[H] High** | Strong competence | Solid understanding, some hands-on, know key concepts and trade-offs |
| **[M] Medium** | Working knowledge | Focused study, understand when/why to use, can discuss intelligently |
| **[L] Low** | Familiarity | Quick pass, high-level intuition, enough for breadth in conversation |
| **[OPTIONAL]** | As needed | Only after completing all core sections |

> [!NOTE]
> Time investment depends on topic complexity and breadth, not just priority level. A medium-priority broad topic (e.g., Unsupervised Learning) may require more hours than a critical but narrow topic.
