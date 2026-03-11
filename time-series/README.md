# Time Series & Forecasting Study Materials

> **Section 4** of the AS Study Agenda | **Total Time**: ~51 hours | **Priority**: Critical for forecasting roles across domains

---

## Study Guide Structure

| File | Section | Topics | Hours | Priority |
|------|---------|--------|-------|----------|
| [01_fundamentals.md](./01_fundamentals.md) | 4.1 | Components, Decomposition, Stationarity | 8h | [C] Critical |
| 02_classical_methods.md | 4.2 | ARIMA, Exponential Smoothing | 6h | [H] High |
| 03_modern_methods.md | 4.3 | Prophet, ML Forecasting, Neural | 9h | [H] High |
| 04_practical.md | 4.4 | Metrics, CV, Seasonalities, Hierarchical | 8h | [H/M] |
| 05_uncertainty.md | 4.5 | Prediction Intervals | 2h | [M] Medium |
| 05_as_critical_topics.md | 4.6 | Cold Start, Sparse Signals, Multi-Step, Calibration, Monitoring, Business | 18h | [H/C] Critical |
| [model_comparison_guide.md](./model_comparison_guide.md) | Cross-cutting | All models: comparison tables, decision flowcharts, phenomenon-based lookup | Ongoing | Reference |

---

## Recommended Study Order

```
Week 1: Foundation
────────────────────────────────────
Day 1-2: 01_fundamentals.md [C]
   └── Components, Decomposition, Stationarity

Day 3-4: 02_classical_methods.md [H]
   └── ARIMA (ACF/PACF, fitting, diagnostics)
   └── Exponential Smoothing (SES, Holt, Holt-Winters)

Day 5: 03_modern_methods.md (Part 1) [H]
   └── Prophet (model, parameters, holidays)


Week 2: Modern Methods & Evaluation
────────────────────────────────────
Day 6: 03_modern_methods.md (Part 2) [H]
   └── ML-based forecasting (XGBoost, feature engineering)

Day 7: 03_modern_methods.md (Part 3) [L]
   └── Neural forecasters (awareness only)

Day 8: 04_practical.md (Part 1) [C]
   └── Evaluation metrics (MAPE, RMSE, WAPE)

Day 9: 04_practical.md (Part 2) [H/M]
   └── Cross-validation, Multiple seasonalities

Day 10: 04_practical.md (Part 3) + 05_uncertainty.md [M]
   └── Hierarchical, Prediction intervals


Week 3: AS-Critical Topics
────────────────────────────────────
Day 11: 05_as_critical_topics.md (Part 1) [H]
   └── Cold Start, Intermittent / Sparse Signals

Day 12: 05_as_critical_topics.md (Part 2) [H]
   └── External Regressors, Ensembles

Day 13: 05_as_critical_topics.md (Part 3) [M]
   └── Anomaly Detection, Missing Data

Day 14: 05_as_critical_topics.md (Part 4) [H/C]
   └── Monitoring, Business Framing

Day 15: 05_as_critical_topics.md (Part 5) [H/M]
   └── Multi-Step Strategy, Forecast Calibration
```

---

## Key Interview Questions by Section

### 4.1 Fundamentals
- "What are the components of a time series?"
- "Additive vs Multiplicative decomposition - when to use each?"
- "What is stationarity and why does it matter?"

### 4.2 Classical Methods
- "Walk me through fitting an ARIMA model"
- "How do you interpret ACF/PACF plots?"
- "ARIMA vs Exponential Smoothing - when to use each?"

### 4.3 Modern Methods
- "When would you use Prophet vs ARIMA?"
- "How do you build an XGBoost forecasting model?"
- "Would you use deep learning for forecasting? When?"

### 4.4 Practical
- "What metric would you use when data has zeros?"
- "How do you cross-validate time series models?"
- "Your data has daily, weekly, and yearly patterns. What approach?"

### 4.6 AS-Critical
- "How would you forecast for a new entity with no history?"
- "Your time series has lots of zeros. What approach?"
- "How do you incorporate external factors into your forecast?"
- "A stakeholder wants a demand forecast. What questions do you ask?"
- "Recursive vs direct multi-step forecasting — when to use each?"
- "How do you know your prediction intervals are well-calibrated?"

---

## Quick Reference

### Depth Levels
| Level | Meaning | Study Approach |
|-------|---------|----------------|
| **[C]** | Critical | Master completely, can teach |
| **[H]** | High | Strong competence, answer confidently |
| **[M]** | Medium | Working knowledge, can discuss |
| **[L]** | Low | Awareness only, know when to look up |

### Files in This Folder
```
time-series/
├── README.md                 (this file)
├── 01_fundamentals.md        [C] Components, Stationarity
├── 02_classical_methods.md   [H] ARIMA, ETS
├── 03_modern_methods.md      [H] Prophet, XGBoost, Neural
├── 04_practical.md           [H/M] Metrics, CV, Hierarchical
├── 05_uncertainty.md         [M] Prediction Intervals
├── 05_as_critical_topics.md  [H/C] Cold Start, Multi-Step, Calibration, Monitoring, etc.
└── model_comparison_guide.md [Ref] Cross-cutting model comparison
```

---

*Parent: [Study Plan](../time_series_study_plan.md) | [Main Agenda](../as_study_agenda.md)*
