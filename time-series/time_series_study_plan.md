---
# Document Outline
- [Executive Summary](#executive-summary)
- [Depth Framework](#depth-framework)
- [Section 4.1: Fundamentals](#41-fundamentals)
  - [4.1.1 Components](#411-time-series-components-c)
  - [4.1.2 Stationarity](#412-stationarity-c)
  - [4.1.3 Autocorrelation & White Noise](#413-autocorrelation--white-noise-h) *(NEW)*
- [Section 4.2: Classical Methods](#42-classical-methods)
- [Section 4.3: Modern Methods](#43-modern-methods)
- [Section 4.4: Practical Considerations](#44-practical-considerations)
- [Section 4.5: Uncertainty Quantification](#45-uncertainty-quantification)
- [Section 4.6: AS-Critical Topics](#46-as-critical-topics-new) *(NEW)*
- [Study Schedule](#study-schedule)
- [Capstone Notebook](#capstone-notebook-end-to-end-forecasting-project) *(NEW)*
- [Completion Checklist](#completion-checklist)

# Executive Summary
Detailed study plan for Time Series & Forecasting (Section 4) using the 4-level depth framework. Includes 8 additional AS-critical topics for supply chain/demand forecasting interviews, plus a capstone notebook project. Total: ~52 hours.
---

# Time Series & Forecasting - Detailed Study Plan

> **Timeline**: Weeks 1-3 | **Hours**: ~52h | **Goal**: Interview-ready for forecasting questions + GitHub portfolio piece

---

## Depth Framework

> Matches the main study agenda (as_study_agenda.md)

| Priority | Depth | Study Approach | Hours/Topic | Stop When... |
|----------|-------|----------------|-------------|--------------|
| **[C] Critical** | Mastery | Deep study, hands-on, practice explaining. Must be able to teach others. | 4-6h | Can explain to a non-expert AND answer follow-up questions |
| **[H] High** | Strong competence | Solid understanding, some hands-on. Know key concepts and trade-offs. | 2-4h | Can answer interview questions confidently |
| **[M] Medium** | Working knowledge | Focused study. Understand when/why to use. Can discuss intelligently. | 1-2h | Can discuss in conversation and know when to apply |
| **[L] Low** | Familiarity | Quick pass, high-level intuition. Enough for breadth in conversation. | 30min-1h | Can mention it and know when to look it up |

> [!TIP]
> **Avoiding Overstudy**: Once you hit the learning objectives for your depth level, STOP and move on.
> **Avoiding Understudy**: Don't move on until you can pass the "Completion Test" for that topic.

---

## 4.1 Fundamentals [C]

### 4.1.1 Time Series Components [C]

| Priority | **[C] Critical** |
|----------|------------------|
| Hours | 4h |
| Approach | Master completely. Practice explaining decomposition to others. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Given a plot, identify and label: trend, seasonality, cycles, noise | [ ] |
| 2 | Explain additive vs multiplicative decomposition with real examples | [ ] |
| 3 | Write the formulas: `Y = T + S + R` and `Y = T × S × R` | [ ] |
| 4 | Apply STL decomposition in Python, interpret each component | [ ] |
| 5 | Explain when multiplicative is preferred (seasonal variation scales with level) | [ ] |
| 6 | Teach this concept to someone in 2 minutes | [ ] |

**Completion Test**:
> *"A retail store's seasonal swings are larger in December when sales are higher. What decomposition and why?"*

**You're ready to move on when**: You can answer the completion test AND explain decomposition without notes.

---

### 4.1.2 Stationarity [C]

| Priority | **[C] Critical** |
|----------|------------------|
| Hours | 4h |
| Approach | Master testing and transformations. Must explain why it matters. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Define stationarity: constant mean, constant variance, autocorrelation depends only on lag | [ ] |
| 2 | Explain why stationarity matters for forecasting (most methods assume it) | [ ] |
| 3 | Perform ADF test: "p < 0.05 → reject null → series IS stationary" | [ ] |
| 4 | Perform KPSS test: "p < 0.05 → reject null → series is NOT stationary" | [ ] |
| 5 | Apply differencing to achieve stationarity | [ ] |
| 6 | Handle the case when ADF and KPSS disagree (trend-stationarity) | [ ] |

**Completion Test**:
> *"ADF says p=0.03 but KPSS says p=0.02. What's happening and what do you do?"*

**You're ready to move on when**: You can test stationarity, interpret conflicting results, and transform data appropriately.

---

### 4.1.3 Autocorrelation & White Noise [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 3h |
| Approach | Foundation for ARIMA. Must understand before fitting models. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Define autocorrelation: "Correlation of a series with its lagged self" | [ ] |
| 2 | Define white noise: mean=0, constant variance, zero autocorrelation at all lags | [ ] |
| 3 | Explain ACF: correlation at each lag k | [ ] |
| 4 | Explain PACF: direct correlation at lag k, controlling for shorter lags | [ ] |
| 5 | Identify AR(p) signature: ACF tails off gradually, PACF cuts off after lag p | [ ] |
| 6 | Identify MA(q) signature: ACF cuts off after lag q, PACF tails off gradually | [ ] |
| 7 | Perform Ljung-Box test: H₀ = no autocorrelation (want p > 0.05 for residuals) | [ ] |
| 8 | Explain random walk: Yₜ = Yₜ₋₁ + εₜ, has unit root, NOT mean-reverting | [ ] |
| 9 | Explain why random walks aren't forecastable: best prediction = last value | [ ] |

**Completion Test**:
> *"Your model's residuals show significant autocorrelation at lag 1. What does this mean and what do you do?"*

**You're ready to move on when**: You can interpret ACF/PACF plots and diagnose residuals with Ljung-Box.

---

## 4.2 Classical Methods [H]

### 4.2.1 ARIMA Family [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 5h |
| Approach | Solid understanding. Know how to fit, diagnose, and explain trade-offs. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain AR(p) in plain English: "Today's value depends on p past values" | [ ] |
| 2 | Explain MA(q) in plain English: "Today's value depends on q past errors" | [ ] |
| 3 | Write AR(1) formula: `Y_t = c + φ₁Y_{t-1} + ε_t` | [ ] |
| 4 | Interpret ACF/PACF to identify p and q | [ ] |
| 5 | Explain what "I" (integrated) means: differencing for non-stationarity | [ ] |
| 6 | Fit ARIMA in Python and check residuals for white noise | [ ] |
| 7 | Use AIC/BIC for model selection | [ ] |
| 8 | Explain SARIMA: "ARIMA + seasonal terms (P, D, Q, m)" | [ ] |
| 9 | **[NEW]** Explain ARIMAX: "ARIMA with external regressors (promotions, price, weather)" | [ ] |
| 10 | **[NEW]** Use `pmdarima.auto_arima()` for automated order selection | [ ] |
| 11 | **[NEW]** Know when to trust auto-selection vs. manual Box-Jenkins | [ ] |

**Completion Test**:
> *"Walk me through fitting an ARIMA model to a new dataset."*
> *"How would you incorporate promotional effects into your forecast?"*

**You're ready to move on when**: You can fit ARIMA/ARIMAX, use auto-selection appropriately, and explain your choices confidently.

---

### 4.2.2 Exponential Smoothing [M]

| Priority | **[M] Medium** |
|----------|----------------|
| Hours | 2h |
| Approach | Working knowledge. Know when to use and key differences from ARIMA. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain SES: "Weighted average, recent values weighted more" | [ ] |
| 2 | Know the progression: SES → Holt (trend) → Holt-Winters (seasonality) | [ ] |
| 3 | Understand ETS notation: Error-Trend-Seasonal (None/Additive/Multiplicative) | [ ] |
| 4 | Know when to use: "Good for short-term, simple patterns" | [ ] |

**Completion Test**:
> *"What's the difference between ARIMA and Exponential Smoothing?"*

**You're ready to move on when**: You can explain the difference and know when each is appropriate.

---

### 4.2.3 Vector Autoregression (VAR) [L]

| Priority | **[L] Low** |
|----------|-------------|
| Hours | 1h |
| Approach | Familiarity only. Know it exists and basic concept. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain VAR: "Multivariate ARIMA - each series depends on lags of ALL series" | [ ] |
| 2 | Know use case: cross-series dependencies (price vs demand, supply vs inventory) | [ ] |
| 3 | Know impulse response: "Shock one series, watch effects propagate" | [ ] |
| 4 | Know limitation: curse of dimensionality with many series | [ ] |

**Completion Test**:
> *"When would VAR be useful instead of univariate ARIMA?"*

**You're ready to move on when**: You can recognize when multivariate modeling is appropriate.

---

## 4.3 Modern Methods [H]

### 4.3.1 Prophet [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 4h |
| Approach | Strong competence. Know the model, key parameters, and trade-offs. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Write Prophet model: `y(t) = g(t) + s(t) + h(t) + ε(t)` | [ ] |
| 2 | Explain each component: g(trend), s(seasonality), h(holidays) | [ ] |
| 3 | Know key parameters: `changepoint_prior_scale`, `seasonality_prior_scale` | [ ] |
| 4 | Explain changepoint detection: how Prophet finds trend changes automatically | [ ] |
| 5 | Add custom holidays and seasonality | [ ] |
| 6 | Perform cross-validation with Prophet's built-in tools | [ ] |
| 7 | Know limitations: needs 2+ cycles, not for high-frequency data | [ ] |

**Completion Test**:
> *"When would you use Prophet instead of ARIMA?"*

**You're ready to move on when**: You can fit Prophet, tune it, and justify when to use it.

---

### 4.3.2 ML-Based Forecasting [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 4h |
| Approach | Strong competence. Know feature engineering and avoid common pitfalls. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | List essential features: lags, rolling stats, date features, Fourier terms | [ ] |
| 2 | Explain data leakage in time series and how to prevent it | [ ] |
| 3 | Build XGBoost/LightGBM forecasting pipeline from scratch | [ ] |
| 4 | Explain why random train/test split is WRONG for time series | [ ] |
| 5 | Know multi-step approaches: recursive vs direct | [ ] |
| 6 | Explain **Global vs Local models**: one model across all series vs one model per series | [ ] |
| 7 | Know when global wins: many series, cross-learning, cold start | [ ] |

**Completion Test**:
> *"How would you use XGBoost/LightGBM for a 7-day ahead forecast?"*

**You're ready to move on when**: You can build a correct pipeline without leakage issues.

---

### 4.3.3 Neural Forecasters [L]

| Priority | **[L] Low** |
|----------|-------------|
| Hours | 1h |
| Approach | Familiarity only. Know what exists and when to consider it. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Know N-BEATS exists: "Fully connected NN, interpretable stacks" | [ ] |
| 2 | Know TFT exists: "Transformer-based, multiple inputs, attention" | [ ] |
| 3 | Know **DeepAR** (Amazon): "Probabilistic forecasting with RNNs, handles scale" | [ ] |
| 4 | Know **Foundation Models** (Chronos/TimesFM): "Zero-shot forecasting, pre-trained on huge data" | [ ] |
| 5 | Explain **Global vs Local**: "Cross-learning from many series vs one model per series" | [ ] |
| 6 | Know when to consider DL: large data (10k+ series), complex dependencies | [ ] |
| 7 | Know when NOT to use: small data, fast training needed, interpretability | [ ] |

**Completion Test**:
> *"Would you use deep learning for forecasting a single product's weekly sales?"*

**You're ready to move on when**: You can answer "probably not, here's why" and know when DL makes sense.

---

## 4.4 Practical Considerations [H]

### 4.4.1 Evaluation Metrics [C]

| Priority | **[C] Critical** |
|----------|------------------|
| Hours | 3h |
| Approach | Master completely. Must know when to use each and common pitfalls. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Calculate by hand: MAE, RMSE, MAPE, SMAPE | [ ] |
| 2 | Write formulas from memory | [ ] |
| 3 | Know MAPE's problems: undefined at zero, asymmetric, ignores volume | [ ] |
| 4 | Explain when to use each metric (see table below) | [ ] |
| 5 | Explain WAPE and why it's better for business aggregations | [ ] |
| 6 | Explain MASE: scale-free metric using naive forecast as baseline | [ ] |
| 7 | Diagnose why a "good" MAPE might not satisfy stakeholders | [ ] |

| Metric | Use When | Avoid When |
|--------|----------|------------|
| MAE | All errors equally costly | Need scale-free comparison |
| RMSE | Large errors especially bad | Outliers present |
| MAPE | Comparing across scales | Data has zeros |
| MASE | Scale-free comparison, intermittent data | Need business-intuitive % |
| WAPE | Business aggregations | Per-item accuracy needed |

**Completion Test**:
> *"Your MAPE is 12% but the business is unhappy. What went wrong?"*

**You're ready to move on when**: You can choose and justify metrics for any business context.

---

### 4.4.2 Cross-Validation for Time Series [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 2h |
| Approach | Strong understanding of correct validation approaches. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain why k-fold CV is wrong for time series | [ ] |
| 2 | Implement walk-forward validation | [ ] |
| 3 | Explain expanding vs sliding window trade-offs | [ ] |
| 4 | Handle multi-step horizons in CV | [ ] |

**Completion Test**:
> *"How would you cross-validate a time series model?"*

**You're ready to move on when**: You can implement correct CV and explain why standard CV fails.

---

### 4.4.3 Multiple Seasonalities [M]

| Priority | **[M] Medium** |
|----------|----------------|
| Hours | 2h |
| Approach | Working knowledge. Know approaches and their trade-offs. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Give examples: daily + weekly + yearly patterns | [ ] |
| 2 | Know SARIMA handles only ONE seasonal period | [ ] |
| 3 | Know approaches for multiple seasonalities: **MSTL**, Prophet, TBATS, Fourier features | [ ] |
| 4 | Explain **MSTL**: STL extended for multiple seasonal periods | [ ] |
| 5 | Explain Fourier terms: sin/cos at different frequencies | [ ] |
| 6 | Know Prophet handles multiple seasonalities automatically via Fourier terms | [ ] |

**Completion Test**:
> *"Your data has daily, weekly, and yearly patterns. What can handle this?"*

**You're ready to move on when**: You can list viable approaches (MSTL, Prophet, TBATS) and explain why SARIMA can't.

---

### 4.4.4 Hierarchical Forecasting [M]

| Priority | **[M] Medium** |
|----------|----------------|
| Hours | 1h |
| Approach | Working knowledge. Know it exists and basic approaches. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain what hierarchical forecasting is | [ ] |
| 2 | Know three approaches: top-down, bottom-up, middle-out | [ ] |
| 3 | Know reconciliation exists (makes forecasts coherent) | [ ] |

**Completion Test**:
> *"You forecast 1000 stores and need company-level. What approaches exist?"*

**You're ready to move on when**: You can describe the approaches at a high level.

---

## 4.5 Uncertainty Quantification [M]

### 4.5.1 Prediction Intervals [M]

| Priority | **[M] Medium** |
|----------|----------------|
| Hours | 2h |
| Approach | Working knowledge. Know why and how to generate intervals. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain why point forecasts are insufficient | [ ] |
| 2 | Know three approaches: parametric, bootstrapping, conformal | [ ] |
| 3 | Explain parametric: assumes normal errors, uses formula | [ ] |
| 4 | Explain bootstrapping: resample residuals, take quantiles | [ ] |
| 5 | Know conformal prediction basics: distribution-free, guaranteed coverage [L] | [ ] |
| 6 | Generate intervals for XGBoost using quantile regression | [ ] |

**Completion Test**:
> *"How would you generate prediction intervals for XGBoost?"*

**You're ready to move on when**: You can explain 2+ methods for generating intervals.

---

## 4.6 AS-Critical Topics [NEW]

> These topics are frequently asked in Applied Scientist interviews for supply chain/demand roles but often missing from standard curricula.

### 4.6.1 Cold Start / New Product Forecasting [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 2h |
| Approach | Critical interview topic. Know multiple strategies and trade-offs. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain the cold start problem: "No historical data for new products" | [ ] |
| 2 | List approaches: similar product matching, attribute-based models, category averages | [ ] |
| 3 | Explain how to use product attributes (category, price, size) to predict demand | [ ] |
| 4 | Know the trade-off: more similar products = better match but smaller sample | [ ] |
| 5 | Discuss warm-up period: how long until enough data for direct forecasting | [ ] |

**Completion Test**:
> *"We're launching a new product next month. How would you forecast its demand?"*

**You're ready to move on when**: You can propose 3+ strategies with trade-offs.

---

### 4.6.2 Intermittent Demand [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 2h |
| Approach | Critical for spare parts, B2B, low-volume SKUs. Common supply chain interview topic. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain intermittent demand: "Many zeros, sporadic non-zero values" | [ ] |
| 2 | Know why standard methods fail (assume continuous demand) | [ ] |
| 3 | Explain Croston's method: separate models for demand size and timing | [ ] |
| 4 | Know SBA (Syntetos-Boylan Approximation) as improvement to Croston | [ ] |
| 5 | Know when to use: spare parts, slow-moving inventory, B2B | [ ] |
| 6 | Understand alternative: aggregate to longer time periods | [ ] |

**Completion Test**:
> *"Our spare parts data is mostly zeros with occasional demand spikes. What approach would you use?"*

**You're ready to move on when**: You can explain Croston's and when standard methods fail.

---

### 4.6.3 External Regressors [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 2h |
| Approach | Essential for real-world forecasting. Interviews often ask about incorporating external factors. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | List common external regressors: promotions, price, weather, holidays, events | [ ] |
| 2 | Explain the difference: known future vs unknown future regressors | [ ] |
| 3 | Know how to incorporate in ARIMAX (ARIMA with exogenous variables) | [ ] |
| 4 | Know how to add regressors in Prophet | [ ] |
| 5 | Explain the challenge: need to forecast regressors or have future values | [ ] |
| 6 | Discuss causal interpretation: correlation vs causation for regressors | [ ] |

**Completion Test**:
> *"How would you incorporate promotional effects into your demand forecast?"*

**You're ready to move on when**: You can add promotions to a model and discuss causality concerns.

---

### 4.6.4 Forecast Ensembles & Combinations [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 2h |
| Approach | Common interview question: "How would you improve your forecast?" |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain why ensembles often outperform single models (reduces variance) | [ ] |
| 2 | Know simple averaging: often surprisingly effective | [ ] |
| 3 | Know weighted averaging: based on validation performance | [ ] |
| 4 | Explain stacking: using a meta-model to combine forecasts | [ ] |
| 5 | Discuss the M-competition findings: combinations often win | [ ] |
| 6 | Know when NOT to ensemble: added complexity, interpretation, latency | [ ] |

**Completion Test**:
> *"Your Prophet model has 15% MAPE. How would you try to improve it?"*

**You're ready to move on when**: You can propose ensemble strategies and cite M-competition.

---

### 4.6.5 Anomaly Detection in Time Series [M]

| Priority | **[M] Medium** |
|----------|----------------|
| Hours | 1.5h |
| Approach | Working knowledge. Know methods and when needed. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain why anomaly detection matters: data quality, event detection | [ ] |
| 2 | Know statistical methods: z-score on residuals, IQR | [ ] |
| 3 | Know model-based: unusual residuals from fitted model | [ ] |
| 4 | Distinguish: point anomalies vs contextual vs collective | [ ] |
| 5 | Know what to do with anomalies: remove, impute, keep, flag | [ ] |

**Completion Test**:
> *"How would you detect and handle anomalies in your training data?"*

**You're ready to move on when**: You can describe 2+ detection methods and handling strategies.

---

### 4.6.6 Missing Data in Time Series [M]

| Priority | **[M] Medium** |
|----------|----------------|
| Hours | 1.5h |
| Approach | Working knowledge. Real data is messy. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain types: random missing vs systematic (e.g., store closures) | [ ] |
| 2 | Know simple imputation: forward fill, backward fill, linear interpolation | [ ] |
| 3 | Know model-based imputation: use seasonal patterns | [ ] |
| 4 | Discuss the problem: imputed values shouldn't be treated as real for training | [ ] |
| 5 | Know when to drop vs impute: amount missing, pattern, downstream use | [ ] |

**Completion Test**:
> *"Your sales data has 5% missing values. What's your approach?"*

**You're ready to move on when**: You can choose an imputation strategy based on missingness pattern.

---

### 4.6.7 Model Monitoring & Retraining [H]

| Priority | **[H] High** |
|----------|--------------|
| Hours | 2h |
| Approach | Critical for production systems. Shows senior-level thinking. |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Explain why monitoring is needed: concept drift, data drift | [ ] |
| 2 | Know metrics to monitor: forecast error over time, bias | [ ] |
| 3 | Explain drift detection: compare recent error to baseline | [ ] |
| 4 | Know retraining strategies: scheduled vs triggered | [ ] |
| 5 | Discuss the trade-off: retrain too often (noise) vs too rarely (drift) | [ ] |
| 6 | Know practical approaches: rolling window, expanding window | [ ] |

**Completion Test**:
> *"How would you know when your production forecast model needs retraining?"*

**You're ready to move on when**: You can design a monitoring and retraining strategy.

---

### 4.6.8 Business Framing for Forecasting [C]

| Priority | **[C] Critical** |
|----------|------------------|
| Hours | 2h |
| Approach | Master this. Most AS interviews start with "tell me about a forecasting problem you've solved." |

**Learning Objectives**:

| # | Objective | Check |
|---|-----------|-------|
| 1 | Frame a business problem as a forecasting problem | [ ] |
| 2 | Identify the decision the forecast supports (inventory, staffing, budget) | [ ] |
| 3 | Choose appropriate forecast horizon based on decision lead time | [ ] |
| 4 | Choose appropriate granularity (hourly, daily, weekly, by store, by SKU) | [ ] |
| 5 | Define success metrics aligned with business impact, not just accuracy | [ ] |
| 6 | Explain trade-offs in overprediction vs underprediction (asymmetric costs) | [ ] |
| 7 | Present results to non-technical stakeholders | [ ] |

**Completion Test**:
> *"A retailer wants to forecast demand. What questions would you ask before building a model?"*

**You're ready to move on when**: You can lead a discovery conversation and translate business needs to technical specs.

---

## Study Schedule

| Day | Topic | Priority | Hours | Completion Signal |
|-----|-------|----------|-------|-------------------|
| 0.5 | 4.6.8 Business Framing | [C] | 2h | Can frame business problems as forecasting |
| 1 | 4.1.1 Components | [C] | 4h | Can teach decomposition |
| 2 | 4.1.2 Stationarity | [C] | 4h | Can test and transform |
| 2.5 | 4.1.3 Autocorrelation & White Noise | [H] | 3h | Can interpret ACF/PACF |
| 3 | 4.2.1 ARIMA | [H] | 4h | Can fit and explain |
| 4 | 4.2.2 ETS | [M] | 2h | Know differences from ARIMA |
| 5 | 4.3.1 Prophet | [H] | 4h | Can fit with holidays |
| 6 | 4.3.2 ML Forecasting | [H] | 4h | Can build pipeline |
| 7 | 4.3.3 Neural | [L] | 1h | Know when (not) to use |
| 8 | 4.4.1 Metrics | [C] | 3h | Can choose and justify |
| 9 | 4.4.2 CV + 4.4.3 Seasonalities | [H]/[M] | 4h | Can validate correctly |
| 10 | 4.4.4 Hierarchical + 4.5.1 Intervals | [M] | 3h | Working knowledge |
| 11 | 4.6.1 Cold Start + 4.6.2 Intermittent | [H] | 4h | Know special demand cases |
| 12 | 4.6.3 External Regressors + 4.6.4 Ensembles | [H] | 4h | Can incorporate external factors |
| 13 | 4.6.5 Anomaly + 4.6.6 Missing Data | [M] | 3h | Can handle dirty data |
| 14 | 4.6.7 Monitoring | [H] | 2h | Can design monitoring & retraining strategy |
| 15 | **Capstone Notebook** | [C] | 4h | End-to-end forecasting project on GitHub |

**Total**: ~52 hours (~3 weeks at 15-20h/week)

---

## Summary by Depth

| Priority | Topics | Total Hours |
|----------|--------|-------------|
| **[C] Critical** | Business Framing, Components, Stationarity, Metrics, Capstone Notebook | 17h |
| **[H] High** | Autocorrelation & White Noise, ARIMA, Prophet, ML Forecasting, CV, Cold Start, Intermittent, External Regressors, Ensembles, Monitoring | 25h |
| **[M] Medium** | ETS, Seasonalities, Hierarchical, Intervals, Anomaly Detection, Missing Data | 10h |
| **[L] Low** | Neural Forecasters | 1h |
| **TOTAL** | 21 topics + capstone | ~52h |

---

## Capstone Notebook: End-to-End Forecasting Project

> **Hours**: 4h | **Priority**: [C] Critical | **Day**: 15
> This is the tangible deliverable from Phase 0. Push to GitHub as a portfolio piece.

### Objective
Build a single, polished notebook that demonstrates the full forecasting workflow — from business framing to model comparison to uncertainty quantification.

### Dataset Options (pick one)
- **Kaggle Store Sales** — retail demand with promotions, holidays, oil prices
- **M5 Competition** — Walmart hierarchical sales data
- **Any real dataset with seasonality, external regressors, and enough history**

### Notebook Structure

| Section | What to Include | Demonstrates |
|---------|----------------|-------------|
| 1. Business Framing | What decision does this forecast support? Horizon, granularity, success metric justification | 4.6.8 |
| 2. EDA & Decomposition | Trend/seasonality identification, stationarity tests (ADF/KPSS), ACF/PACF | 4.1.1-4.1.3 |
| 3. Baseline | Naive and seasonal naive forecasts | 4.4.1 (MASE baseline) |
| 4. Model 1: ARIMA/SARIMA | Fit, diagnose residuals, forecast | 4.2.1 |
| 5. Model 2: Prophet | With holidays and external regressors | 4.3.1, 4.6.3 |
| 6. Model 3: XGBoost | Lag features, rolling stats, proper train/test split | 4.3.2 |
| 7. Model Comparison | Walk-forward CV, compare MASE/MAPE/RMSE in a table | 4.4.1-4.4.2 |
| 8. Prediction Intervals | Quantile regression or bootstrap intervals for best model | 4.5.1 |
| 9. Conclusions | Which model wins and why, limitations, what you'd do in production | 4.6.7, 4.6.8 |

### Completion Criteria
- [ ] Notebook runs end-to-end without errors
- [ ] At least 3 models compared with proper time series CV (no data leakage)
- [ ] Metrics table with MASE, MAPE, RMSE for all models
- [ ] Prediction intervals generated for the best model
- [ ] Business framing section answers: what decision, what horizon, what metric, why
- [ ] Pushed to GitHub with a clear README

### Interview Story (practice this)
> *"I built an end-to-end demand forecasting pipeline comparing ARIMA, Prophet, and XGBoost on [dataset]. I used walk-forward cross-validation and found that [model] performed best with a MASE of [X]. The key insight was [insight about seasonality/regressors/etc]. I also generated prediction intervals using [method] to quantify uncertainty for inventory decisions."*

---

## Completion Checklist

### Week 1 Checkpoint (Days 0.5-5)
- [ ] Can answer: "A retailer wants demand forecasting. What questions do you ask?" [C]
- [ ] Can answer: "What is stationarity and why does it matter?" [C]
- [ ] Can answer: "Walk me through fitting an ARIMA model" [H]
- [ ] Can answer: "When would you use Prophet vs ARIMA?" [H]
- [ ] Can decompose a time series and interpret components [C]

### Week 2 Checkpoint (Days 6-10)
- [ ] Can answer: "How do you build an XGBoost forecasting model?" [H]
- [ ] Can answer: "What metric would you use when data has zeros?" [C]
- [ ] Can answer: "How do you cross-validate time series models?" [H]
- [ ] Can generate prediction intervals [M]

### Week 3 Checkpoint (Days 11-15) - AS-Critical + Capstone
- [ ] Can answer: "How would you forecast demand for a new product?" [H]
- [ ] Can answer: "Your data has many zeros. What approach?" [H]
- [ ] Can answer: "How would you incorporate promotions into your forecast?" [H]
- [ ] Can answer: "How would you know when to retrain your model?" [H]
- [ ] Capstone notebook complete and pushed to GitHub [C]

### Final Verification
All topics should have their Learning Objectives checked off before moving to the next section.

