# Battle Plan - Phase 0: Classical ML Foundation (WEEK 0.5-1.5)

**Duration**: 2 weeks (before Month 1)  
**Total Hours**: 24 hours  
**Purpose**: Fill critical gaps in Classical ML (ARIMA, HMMs, EM, ICA, Survival Analysis)

---

## 📋 OVERVIEW

This phase was ADDED based on 100% gap analysis of 1,023 Applied Scientist interview questions.

**Critical Gaps Identified**:
- ARIMA time series (43 mentions, 0% coverage)
- Hidden Markov Models (7 mentions, 0% coverage)
- **ICA** - Independent Component Analysis (230 mentions!, 0% coverage)
- EM Algorithm / GMMs (6 mentions, 0% coverage)
- Survival Analysis (multiple phases, 0% coverage)

**Study Materials**: See `study_materials_guide_added_modules.md` for all textbook chapters

---

## WEEK 0.5: TIME SERIES & FORECASTING (12 hours)

### **Day 1-2: ARIMA Foundations** (6h)

**Learning Objectives**:
- Derive ARIMA from first principles (understand why p, d, q parameters exist)
- Build intuition for stationarity vs non-stationarity
- Master ACF/PACF interpretation visually

**Hour-by-Hour Breakdown**:

**Hour 1-2: Stationarity Deep Dive**
- Theory: Weak vs strong stationarity, why it matters
- Visual intuition: Plot stationary vs non-stationary series
- **Hands-on**: Generate synthetic data with `numpy`, apply ADF test
- **Exercise**: Identify which real-world series need differencing

**Hour 3-4: AR, MA, ARMA Models**
- Theory: Autoregressive (AR) process - derive from lag operator
- Moving Average (MA) process - why use past errors?
- ACF/PACF patterns (memorize: AR → ACF decays, PACF cuts; MA opposite)
- **Hands-on**: Implement AR(2) from scratch in NumPy
- **Exercise**: Simulate MA(1), plot ACF/PACF, verify textbook patterns

**Hour 5-6: ARIMA Integration & Model Selection**
- Theory: When to use integrated (I) component - unit root vs differencing
- Model selection: Box-Jenkins methodology step-by-step
- **Hands-on**: `statsmodels.tsa.arima` - fit ARIMA(2,1,2) to airline passengers
- **Exercise**: Use grid search to find optimal (p,d,q) for custom dataset
- **Interview Prep**: Explain ARIMA to a 5-year-old (Feynman technique)

**Resources (Best-in-Class)**:
- **Intuition**: [StatQuest] "ARIMA Models" & "Autocorrelation" (YouTube) - *Best visual guide.*
- **Rigor**: "Forecasting: Principles and Practice" (Hyndman) - Ch 8-9 (Online) - *The standard text.*
- **Code**: `statsmodels` official notebooks.

**Deliverable**: Notebook with 3 ARIMA implementations on different datasets + written explanation

---

### **Day 3: Prophet vs ARIMA** (3h)

**Learning Objectives**:
- Understand when Prophet beats ARIMA (seasonality, holidays, missing data)
- Know Facebook's additive model decomposition

**Hour-by-Hour Breakdown**:

**Hour 1: Prophet Architecture**
- Theory: Trend (piecewise linear/logistic) + Seasonality (Fourier) + Holidays
- Why Bayesian? (Automatic changepoint detection, uncertainty intervals)
- **Hands-on**: Install `prophet`, fit to daily website traffic

**Hour 2: Comparative Analysis**
- **Exercise**: Same dataset, fit both ARIMA & Prophet
- Compare: RMSE, interpretability, handling missing data, speed
- **Hands-on**: Add custom holiday (e.g., Black Friday) to Prophet

**Hour 3: When to Use What?**
- Decision tree: ARIMA vs Prophet vs Exponential Smoothing
- **Interview Prep**: "Design a demand forecasting system for Uber Eats"
- **Written**: 1-page comparison table

**Resources**:
- Prophet documentation + paper (Taylor & Letham, 2017)
- Comparison blog: "ARIMA vs Prophet vs LSTM"

**Deliverable**: Comparative analysis notebook + decision framework document

---

### **Day 4-5: Hidden Markov Models** (6h)

**Learning Objectives**:
- Build from Markov chains → HMMs (why "hidden"?)
- Master Viterbi algorithm (decoding)
- Understand Forward-Backward & Baum-Welch (learning)

**Hour-by-Hour Breakdown**:

**Hour 1-2: Markov Chains to HMMs**
- Theory: Review Markov property, transition matrix
- Hidden states vs observations (weather example: hidden=actual, obs=umbrella)
- Three canonical problems: Evaluation, Decoding, Learning
- **Hands-on**: Implement simple Markov chain (weather model) in NumPy
- **Visual**: Draw state diagram, transition matrix heatmap

**Hour 3-4: Viterbi Algorithm (Decoding)**
- Theory: Dynamic programming for most likely state sequence
- Derive Viterbi recursion on whiteboard/paper
- **Hands-on**: Implement Viterbi from scratch (NumPy only, no libraries)
- **Exercise**: Given observations [Umbrella, No Umbrella, Umbrella], decode weather states
- **Test**: Verify against `hmmlearn` library

**Hour 5: Forward-Backward Algorithm**
- Theory: Compute P(Observations) - why needed? (model comparison)
- Forward pass: α recursion
- Backward pass: β recursion
- **Hands-on**: Implement forward algorithm
- **Exercise**: Calculate likelihood of observation sequence

**Hour 6: Baum-Welch (Learning)**
- Theory: EM for HMMs - E-step (forward-backward), M-step (re-estimate params)
- Why EM? (latent variables = hidden states)
- **Hands-on**: Use `hmmlearn.GaussianHMM` to learn from data
- **Exercise**: Train HMM on stock price data, interpret learned states (bull/bear market)

**Resources (Best-in-Class)**:
- **Intuition**: "Speech and Language Processing" (Jurafsky) - Appendix A (HMMs) - *Readable & practical.*
- **Rigor**: Rabiner, "A Tutorial on HMMs" (1989) - *The derivation bible.*
- **Video**: StatQuest "Hidden Markov Models" - *For the 'why'.*

**Deliverable**: 
1. Viterbi implementation from scratch (tested)
2. Notebook: HMM for regime detection in time series
3. Written: ELI5 explanation of when to use HMMs

---

### **Day 6: Survival Analysis Fundamentals** (3h)

**Learning Objectives**:
- Understand censoring (right, left, interval; MCAR/MAR/MNAR)
- Kaplan-Meier estimator intuition
- Cox Proportional Hazards introduction

**Hour-by-Hour Breakdown**:

**Hour 1: Censoring & Survival Function**
- Theory: Why survival analysis? (time-to-event with incomplete data)
- Censoring types with real examples (customer churn = right-censored)
- Survival function S(t) = P(T > t), Hazard function h(t)
- **Hands-on**: Generate synthetic survival data with censoring

**Hour 2: Kaplan-Meier Estimator**
- Theory: Non-parametric survival curve estimation
- Derive K-M formula (product-limit estimator)
- **Hands-on**: Implement K-M from scratch, plot survival curve
- **Exercise**: Compare survival curves for two groups (log-rank test)
- **Library**: Use `lifelines` library, verify against manual implementation

**Hour 3: Cox Proportional Hazards (Intro)**
- Theory: Semi-parametric model, hazard ratio interpretation
- Proportional hazards assumption - what it means
- **Hands-on**: Fit Cox model with `" lifelines.CoxPHFitter`
- **Exercise**: Customer churn analysis - which features predict churn?
- **Interview Prep**: "How would you model time-to-conversion for ad campaigns?"

**Resources**:
- "Survival Analysis: A Self-Learning Text" (Kleinbaum & Klein) - Ch 1-2
- `lifelines` documentation + examples
- StatQuest YouTube: "Survival Analysis" series

**Deliverable**: Churn prediction notebook using survival analysis

---

## WEEK 1.5: EM ALGORITHM & CLASSICAL METHODS (12 hours)

### **Day 1-2: EM Algorithm & Gaussian Mixture Models** (6h)

**Learning Objectives**:
- Build intuition: why EM? (handle latent variables)
- Derive EM for GMMs from first principles
- Understand convergence properties

**Hour-by-Hour Breakdown**:

**Hour 1-2: EM Theory Foundation**
- **Motivation**: Clustering with probabilistic assignments (soft k-means)
- Latent variable models - what makes them hard?
- Derive lower bound on log-likelihood (Jensen's inequality)
- E-step: Compute posterior P(Z|X, θ)
- M-step: Maximize expected log-likelihood
- **Visual**: Animate EM on 2D Gaussian mixture (μ, Σ updates)

**Hour 3-4: GMM Implementation**
- Theory: GMM specific E-step (responsibilities) and M-step (update μ, Σ, π)
- **Hands-on**: Implement EM for GMM from scratch (NumPy only)
- **Exercise**: Fit 3-component GMM to Iris dataset, visualize clusters
- **Debug**: Common issues (singular covariance, label switching)
- **Test**: Compare against `sklearn.mixture.GaussianMixture`

**Hour 5-6: EM Extensions & Interview Prep**
- Theory: EM for other models (HMMs - Baum-Welch is EM!)
- Convergence: local optima, initialization strategies (k-means++)
- **Exercise**: Run EM with different initializations, compare results
- **Interview Prep**: "Derive EM for a simple coin-flip example with missing data"
- **Application**: Customer segmentation with GMM

**Resources (Best-in-Class)**:
- **Intuition**: StatQuest "The EM Algorithm" - *Visualizes the E/M steps perfectly.*
- **Rigor**: Bishop, "Pattern Recognition and Machine Learning" - Ch 9.2-9.4 - *The gold standard for EM.*
- **Code**: "Python Data Science Handbook" (VanderPlas) - GMM chapter.

**Deliverable**:
1. EM algorithm implemented from scratch (tested)
2. Jupyter notebook: GMM for customer segmentation
3. Written derivation: EM for GMM (include all math steps)

---

### **Day 3-4: PCA vs ICA - Deep Dive** (4h)

**Learning Objectives**:
- Understand difference: **Uncorrelated** (PCA) vs **Independent** (ICA)
- Derive PCA from eigendecomposition
- Build intuition for ICA (cocktail party problem)

**Hour-by-Hour Breakdown**:

**Hour 1-2: PCA Deep Dive**
- Theory: Maximize variance vs minimize reconstruction error (prove equivalence)
- Derive via eigendecomposition of covariance matrix
- Derive via SVD of data matrix
- **Hands-on**: Implement PCA from scratch (eigendecomposition)
- **Visual**: Plot principal components on 2D data, show variance explained
- **Exercise**: PCA on MNIST, visualize first 2 PCs, reconstruct images

**Hour 3: ICA Introduction**
- Theory: Statistical independence vs uncorrelated (counter-example: X, X²)
- Cocktail party problem intuition (blind source separation)
- Non-Gaussianity requirement (Central Limit Theorem reasoning)
- FastICA algorithm overview (maximizing non-Gaussianity via kurtosis/neg-entropy)
- **Hands-on**: Use `sklearn.decomposition.FastICA` on mixed audio signals

**Hour 4: PCA vs ICA - When to Use What?**
- **Exercise**: Generate data where ICA beats PCA (independent sources)
- **Comparison**: Apply both to EEG/fMRI-style data
- **Interview Prep**: "When would you choose ICA over PCA for feature extraction?"
- **Application**: Financial data (ICA for hidden factors)

**Resources**:
- "Independent Component Analysis" (Hyvärinen & Oja paper) - FREE, CRITICAL (230 mentions!)
- Scikit-learn: ICA vs PCA tutorial
- 3Blue1Brown: "Eigenvectors and Eigenvalues" (YouTube)

**Deliverable**: Comparative notebook (PCA vs ICA on 3 datasets) + decision framework

---

### **Day 5-6: SVM, Kernels, & Regression Diagnostics** (2h)

**Learning Objectives**:
- Understand SVM dual formulation, kernel trick intuition
- SVR (Support Vector Regression) for regression tasks
- Regression diagnostics: Heteroscedasticity, Multicollinearity (VIF)

**Hour-by-Hour Breakdown**:

**Hour 1-2: SVM & Kernel Trick**
- Theory: Primal vs dual formulation (Lagrangian, KKT conditions)
- Kernel trick: Why it works (inner product in high-dimensional space)
- Common kernels: Linear, RBF (Gaussian), Polynomial
- **Hands-on**: `sklearn.svm.SVC` with different kernels on non-linear data
- **Visual**: Decision boundary visualization for each kernel
- **Exercise**: Grid search for optimal C and gamma (RBF kernel)

**Resources**:
- "An Introduction to Statistical Learning" (ISLR) - Ch 9 (FREE PDF)

**Deliverable**: SVM kernel comparison notebook

---

## PHASE 0 DELIVERABLES

### Week 0.5 Deliverables:
- [ ] ARIMA implementation (3 datasets)
- [ ] Prophet comparison analysis
- [ ] Viterbi algorithm from scratch
- [ ] HMM user behavior notebook
- [ ] Survival analysis churn prediction

### Week 1.5 Deliverables:
- [ ] EM algorithm from scratch
- [ ] GMM customer segmentation notebook
- [ ] Written EM derivation (all steps)
- [ ] PCA vs ICA comparison (3 datasets) + decision framework
- [ ] SVM kernel comparison notebook

### Success Criteria:
- [ ] Can derive EM algorithm on whiteboard
- [ ] Can implement Viterbi from scratch in interview
- [ ] Can explain when to use ARIMA vs Prophet
- [ ] Understand ICA vs PCA (statistical independence vs uncorrelation)
- [ ] Can identify when to use survival analysis

---

## INTEGRATION WITH MONTH 1

After completing Week 0.5-1.5, proceed to **Month 1 (file: `02_battle_plan_months_1-2.md`)**

**Additions to Month 1 (minimal)**:
- Week 1: Add +2h for MCMC theory (Casella Ch 11), +2h distribution relationships
- Total Month 1: 165h → 169h

---

**Total Phase 0**: 24 hours  
**Weekly Commitment**: 12-15h/week if spreading over 2 weeks, or intensive 24h in 1 week

**Status**: ✅ Ready to start - Begin with ARIMA (Hyndman Ch 8)
