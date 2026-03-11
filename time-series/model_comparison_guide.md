---
# Document Outline
- [Objective & Scope](#objective--scope)
- [How to Use This Guide](#how-to-use-this-guide)
- [Master Comparison Table](#master-comparison-table)
- [Decision Flowchart](#decision-flowchart)
- [1. Classical Statistical Models](#1-classical-statistical-models)
  - [AR / MA / ARMA](#11-ar--ma--arma)
  - [ARIMA / SARIMA](#12-arima--sarima)
  - [ARIMAX / SARIMAX](#13-arimax--sarimax)
  - [Exponential Smoothing / ETS](#14-exponential-smoothing--ets)
  - [Theta Method](#15-theta-method)
- [2. Advanced Statistical Models](#2-advanced-statistical-models)
  - [VAR / VECM](#21-var--vecm)
  - [GARCH Family](#22-garch-family)
  - [ARFIMA](#23-arfima-fractional-integration)
  - [TBATS / BATS](#24-tbats--bats)
  - [BSTS / Structural Time Series](#25-bsts--structural-time-series)
- [3. Regime-Switching & Changepoint Models](#3-regime-switching--changepoint-models)
  - [Markov-Switching (MS-AR)](#31-markov-switching-ms-ar)
  - [TAR / SETAR](#32-tar--setar)
  - [STAR / LSTAR / ESTAR](#33-star--lstar--estar)
  - [Bai-Perron Structural Breaks](#34-bai-perron-structural-breaks)
  - [BOCPD](#35-bayesian-online-changepoint-detection-bocpd)
  - [PELT / CUSUM](#36-pelt--cusum)
- [4. Decomposition-Based Models](#4-decomposition-based-models)
  - [STL / MSTL](#41-stl--mstl)
  - [Prophet](#42-prophet-meta)
- [5. Machine Learning Models](#5-machine-learning-models)
  - [LightGBM / XGBoost](#51-lightgbm--xgboost)
  - [Random Forest](#52-random-forest)
  - [Linear Models (Ridge, Lasso, ElasticNet)](#53-linear-models)
- [6. Deep Learning / Neural Models](#6-deep-learning--neural-models)
  - [DeepAR](#61-deepar-amazon)
  - [N-BEATS / N-HiTS](#62-n-beats--n-hits)
  - [Temporal Fusion Transformer (TFT)](#63-temporal-fusion-transformer-tft)
  - [WaveNet](#64-wavenet)
  - [Informer / Autoformer / PatchTST](#65-informer--autoformer--patchtst)
- [7. Foundation Models](#7-foundation-models)
  - [Chronos](#71-chronos-amazon)
  - [TimesFM](#72-timesfm-google)
  - [Moirai / Lag-Llama](#73-moirai--lag-llama)
- [8. Specialized Methods](#8-specialized-methods)
  - [Croston / SBA / TSB](#81-croston--sba--tsb)
  - [Hierarchical Reconciliation](#82-hierarchical-reconciliation)
  - [FFORMA](#83-fforma)
  - [Ensemble / Stacking](#84-ensemble--stacking)
- [Comparison by Phenomenon](#comparison-by-phenomenon)

# Executive Summary

> This is a **living reference document** for comparing ALL time series and forecasting models.
> The goal is a single source of truth for model selection: given a phenomenon, data characteristics,
> and constraints, which model(s) should you reach for? This file is designed to grow incrementally
> as each model family is studied in depth.

---

# Objective & Scope

## Objective

Provide a **single, comprehensive reference** to compare time series and forecasting models across all families — from classical statistical methods to foundation models. This document answers the question:

> **"Given my data, my phenomenon, and my constraints, which model should I use?"**

## Scope

This guide covers **every major model family** used in time series analysis and forecasting:

| Family | Models | Status |
|--------|--------|--------|
| Classical Statistical | AR, MA, ARMA, ARIMA, SARIMA, ARIMAX, ETS, Theta | Scaffolded |
| Advanced Statistical | VAR, VECM, GARCH, ARFIMA, TBATS, BSTS | Scaffolded |
| Regime-Switching | Markov-Switching, TAR, SETAR, STAR, Bai-Perron, BOCPD, PELT | Scaffolded |
| Decomposition-Based | STL, MSTL, Prophet | Scaffolded |
| Machine Learning | LightGBM, XGBoost, Random Forest, Ridge/Lasso | Scaffolded |
| Deep Learning | DeepAR, N-BEATS, N-HiTS, TFT, WaveNet, Informer, PatchTST | Scaffolded |
| Foundation Models | Chronos, TimesFM, Moirai, Lag-Llama | Scaffolded |
| Specialized | Croston, SBA, TSB, Hierarchical, FFORMA, Ensembles | Scaffolded |

**Status key**: `Scaffolded` = structure in place, to be filled in. `Complete` = fully documented.

## What This Is NOT

- Not a tutorial on how to fit each model (see the numbered study files `01_`–`06_` for that)
- Not a theory reference (formulas live in the study files)
- This is a **decision-support and comparison tool**

---

# How to Use This Guide

1. **Quick lookup**: Jump to the [Master Comparison Table](#master-comparison-table) for a side-by-side view
2. **Phenomenon-based**: Jump to [Comparison by Phenomenon](#comparison-by-phenomenon) to find which model fits your data characteristics
3. **Deep dive**: Click into any model section for its card (strengths, weaknesses, when to use, when to avoid)
4. **Decision support**: Use the [Decision Flowchart](#decision-flowchart) for interview or project model selection

---

# Master Comparison Table

> [!NOTE]
> This table will be filled in progressively as each model is studied. Columns:
> - **Univariate/Multi**: Can it model one series (U) or multiple related series (M)?
> - **Exogenous**: Can it incorporate external covariates?
> - **Probabilistic**: Does it produce prediction intervals natively?
> - **Nonlinear**: Can it capture nonlinear relationships?
> - **Scale**: How well does it handle 1000s+ of series in production?
> - **Interpretability**: How easy is it to explain the forecast?

| Model | Family | Univariate/Multi | Exogenous | Probabilistic | Nonlinear | Scale | Interpretability |
|-------|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **ARIMA** | Classical | U | No | Yes | No | Low | High |
| **SARIMA** | Classical | U | No | Yes | No | Low | High |
| **ARIMAX** | Classical | U | Yes | Yes | No | Low | High |
| **ETS** | Classical | U | No | Yes | No | Low | High |
| **Theta** | Classical | U | No | Yes | No | Low | High |
| **VAR** | Advanced | M | No | Yes | No | Low | Medium |
| **VECM** | Advanced | M | No | Yes | No | Low | Medium |
| **GARCH** | Advanced | U | Optional | Yes | Partial | Low | Medium |
| **ARFIMA** | Advanced | U | No | Yes | No | Low | Medium |
| **TBATS** | Advanced | U | No | Yes | No | Low | Medium |
| **BSTS** | Advanced | U | Yes | Yes | No | Medium | High |
| **Markov-Switching** | Regime | U | Optional | Yes | Yes | Low | Medium |
| **TAR / SETAR** | Regime | U | Threshold var | Yes | Yes | Low | High |
| **STAR / LSTAR** | Regime | U | Transition var | Yes | Yes | Low | Medium |
| **STL + Model** | Decomposition | U | Depends | Depends | No | Medium | High |
| **Prophet** | Decomposition | U | Yes | Yes | Partial | Medium | High |
| **LightGBM** | ML | U/M (global) | Yes | Add-on | Yes | High | Medium |
| **XGBoost** | ML | U/M (global) | Yes | Add-on | Yes | High | Medium |
| **DeepAR** | Neural | M (global) | Yes | Yes | Yes | High | Low |
| **N-BEATS** | Neural | U/M | No | Optional | Yes | High | Medium |
| **N-HiTS** | Neural | U/M | No | Optional | Yes | High | Medium |
| **TFT** | Neural | M (global) | Yes | Yes | Yes | High | Medium |
| **Chronos** | Foundation | U | No | Yes | Yes | High | Low |
| **TimesFM** | Foundation | U | No | Yes | Yes | High | Low |
| **Croston** | Specialized | U | No | Limited | No | Medium | High |

---

# Decision Flowchart

> [!TIP]
> Use this flowchart as a starting point for model selection. Real-world choices often depend on additional factors (data size, latency requirements, team expertise).

```
START: What are your data characteristics?
│
├── How many series?
│   ├── 1-10 series
│   │   └── Classical or Advanced Statistical models
│   ├── 10-1000 series
│   │   └── Prophet, Auto-ARIMA, or ML (LightGBM)
│   └── 1000+ series
│       └── Global models: LightGBM, DeepAR, TFT, Foundation
│
├── Is demand intermittent (many zeros)?
│   └── YES → Croston, SBA, TSB
│
├── Multiple seasonal periods?
│   ├── YES → TBATS, MSTL+ARIMA, Prophet
│   └── NO (single or none) → Continue below
│
├── Regime changes / structural breaks?
│   ├── Recurring regimes → Markov-Switching, HMM
│   ├── Observable driver → TAR, SETAR, STAR
│   └── Permanent breaks → Bai-Perron, PELT
│
├── Volatility clustering?
│   └── YES → GARCH family (often paired with ARIMA for mean)
│
├── External regressors available?
│   ├── Few, known future → ARIMAX, Prophet, BSTS
│   └── Many, complex interactions → LightGBM, XGBoost, TFT
│
├── Need probabilistic forecasts?
│   ├── Parametric → ARIMA, ETS, DeepAR, BSTS
│   └── Nonparametric → Conformal prediction on any model
│
└── Need interpretability?
    ├── HIGH → ARIMA, ETS, Prophet, BSTS, TAR
    ├── MEDIUM → LightGBM (SHAP), TFT (attention)
    └── LOW OK → DeepAR, N-BEATS, Foundation models
```

---

# 1. Classical Statistical Models

## 1.1 AR / MA / ARMA

> **Detailed notes**: [02_classical_methods.md](./02_classical_methods.md#ar-ma-arma-components)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Linear models using past values (AR), past errors (MA), or both (ARMA) |
| **Assumes** | Stationarity, constant variance, linear relationships |
| **Best for** | Stationary series with clear ACF/PACF signatures |
| **Avoid when** | Non-stationary data, nonlinear dynamics, multiple seasonalities |
| **Python** | `statsmodels.tsa.arima.model.ARIMA` |

---

## 1.2 ARIMA / SARIMA

> **Detailed notes**: [02_classical_methods.md](./02_classical_methods.md#arima-adding-integration)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | ARMA + differencing for non-stationary data; SARIMA adds seasonal component |
| **Assumes** | Linearity, constant variance, single seasonality (SARIMA) |
| **Best for** | Univariate series with trend and/or single seasonality |
| **Avoid when** | Multiple seasonalities, regime changes, large-scale forecasting |
| **Key params** | (p,d,q) non-seasonal; (P,D,Q,m) seasonal |
| **Python** | `statsmodels.tsa.statespace.sarimax.SARIMAX` |

---

## 1.3 ARIMAX / SARIMAX

> **Detailed notes**: [02_classical_methods.md](./02_classical_methods.md#arimax-adding-external-regressors-h)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | ARIMA + external regressors (promotions, weather, price) |
| **Assumes** | Same as ARIMA + linear effect of regressors |
| **Best for** | When a few known drivers influence the series |
| **Avoid when** | Many regressors, nonlinear covariate effects, unknown future values |
| **Gotcha** | Must provide future regressor values at forecast time |
| **Python** | `statsmodels.tsa.statespace.sarimax.SARIMAX` with `exog=` |

---

## 1.4 Exponential Smoothing / ETS

> **Detailed notes**: [02_classical_methods.md](./02_classical_methods.md#422-exponential-smoothing)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Weighted average of past values with exponentially decaying weights |
| **Variants** | SES (level), Holt (level+trend), Holt-Winters (level+trend+season) |
| **ETS notation** | (Error, Trend, Season) each = {N, A, M} or {Ad, Md} for damped |
| **Best for** | Short-to-medium horizon, clear level/trend/season, fast baseline |
| **Avoid when** | Complex patterns, need covariates, long memory |
| **Python** | `statsmodels.tsa.holtwinters.ExponentialSmoothing` |

---

## 1.5 Theta Method

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Decomposes series into "theta lines" (modified curvatures), forecasts each |
| **Best for** | M3 competition winner; strong simple baseline |
| **Avoid when** | Complex seasonality, need interpretability of components |
| **Python** | `statsforecast.models.Theta` |
| **Status** | *To be expanded* |

---

# 2. Advanced Statistical Models

## 2.1 VAR / VECM

> **Detailed notes**: [02_classical_methods.md](./02_classical_methods.md#423-vector-autoregression-var-l)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Multivariate AR — each series depends on its own lags AND lags of other series |
| **VECM** | VAR + cointegration (shared long-run equilibrium between non-stationary series) |
| **Best for** | 2-5 related series with causal/feedback relationships |
| **Avoid when** | Many series, no inter-series relationships, high dimensionality |
| **Python** | `statsmodels.tsa.vector_ar.var_model.VAR` |

---

## 2.2 GARCH Family

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Models time-varying variance (volatility clustering) |
| **Variants** | GARCH, EGARCH (asymmetric), GJR-GARCH (leverage effect), TGARCH |
| **Best for** | Financial returns, risk modeling, VaR estimation |
| **Avoid when** | Variance is stable, non-financial data with constant volatility |
| **Note** | Usually paired with ARIMA for the mean equation |
| **Python** | `arch` library |
| **Status** | *To be expanded* |

---

## 2.3 ARFIMA (Fractional Integration)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | ARIMA with fractional d (e.g., d=0.3) — captures long memory |
| **Best for** | Series with slow autocorrelation decay (network traffic, hydrology) |
| **Avoid when** | Standard ARIMA works, or d is near 0 or 1 |
| **Python** | `statsmodels.tsa.arima.model.ARIMA` supports fractional d |
| **Status** | *To be expanded* |

---

## 2.4 TBATS / BATS

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Handles multiple seasonal periods via trigonometric representation |
| **Best for** | Multiple seasonalities (daily + weekly + yearly), high-frequency data |
| **Avoid when** | Single seasonality (SARIMA suffices), need covariates |
| **Python** | `tbats` library |
| **Status** | *To be expanded* |

---

## 2.5 BSTS / Structural Time Series

| Attribute | Detail |
|-----------|--------|
| **One-liner** | State-space model with Bayesian inference — decomposes into level, trend, season, regression |
| **Best for** | Causal impact analysis, incorporating prior knowledge, interpretable components |
| **Avoid when** | No Bayesian expertise, need fast fitting on many series |
| **Python** | `causalimpact`, `tensorflow_probability.sts` |
| **Status** | *To be expanded* |

---

# 3. Regime-Switching & Changepoint Models

## 3.1 Markov-Switching (MS-AR)

> **Detailed notes**: [02_classical_methods.md](./02_classical_methods.md#when-arima-fails) (expandable section)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Jointly estimates latent regime (hidden state) and regime-specific AR parameters |
| **Key insight** | NOT two-step; simultaneous MLE via Hamilton filter |
| **Forecasting** | Probability-weighted blend across regimes (or hard-classify if regime is clear) |
| **Best for** | Recurring regime shifts (bull/bear, expansion/recession) |
| **Avoid when** | Stable series, permanent breaks, observable regime driver |
| **Python** | `statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression` |

---

## 3.2 TAR / SETAR

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Regime determined by observable variable crossing a threshold |
| **SETAR** | Self-Exciting: threshold variable = lagged value of the series itself |
| **Best for** | When you KNOW what drives regimes (VIX, yield spread) |
| **Avoid when** | Regime driver is unknown/latent |
| **Python** | Custom implementation or `tsDyn` (R) |
| **Status** | *To be expanded* |

---

## 3.3 STAR / LSTAR / ESTAR

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Like TAR but with smooth (logistic/exponential) transition between regimes |
| **LSTAR** | Logistic transition — asymmetric regimes (e.g., expansions longer than recessions) |
| **ESTAR** | Exponential transition — symmetric regimes |
| **Best for** | Gradual regime transitions |
| **Python** | Custom implementation or `tsDyn` (R) |
| **Status** | *To be expanded* |

---

## 3.4 Bai-Perron Structural Breaks

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Detects permanent breaks in model parameters + estimates break dates |
| **Best for** | Policy changes, market structure shifts that do NOT revert |
| **Avoid when** | Recurring regimes (use Markov-Switching instead) |
| **Status** | *To be expanded* |

---

## 3.5 Bayesian Online Changepoint Detection (BOCPD)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Real-time detection of regime changes as new data arrives |
| **Best for** | Production monitoring, alerting, streaming data |
| **Status** | *To be expanded* |

---

## 3.6 PELT / CUSUM

| Attribute | Detail |
|-----------|--------|
| **One-liner** | PELT = fast exact changepoint detection; CUSUM = sequential shift detection |
| **Best for** | PELT: offline analysis; CUSUM: quality control, monitoring |
| **Python (PELT)** | `ruptures` library |
| **Status** | *To be expanded* |

---

# 4. Decomposition-Based Models

## 4.1 STL / MSTL

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Seasonal-Trend decomposition using LOESS; MSTL handles multiple seasonalities |
| **Best for** | Pre-processing step: decompose then model residual with ARIMA etc. |
| **Avoid when** | Need a standalone forecaster (STL is decomposition, not forecast) |
| **Python** | `statsmodels.tsa.seasonal.STL`, `statsforecast.models.MSTL` |
| **Status** | *To be expanded* |

---

## 4.2 Prophet (Meta)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Additive regression: trend (with changepoints) + seasonality (Fourier) + holidays + regressors |
| **Best for** | Business forecasting with holidays, analyst-friendly, multiple seasonalities |
| **Avoid when** | Very short series, high-frequency data, need strict statistical inference |
| **Key feature** | Automatic changepoint detection in trend |
| **Python** | `prophet` |
| **Status** | *To be expanded* |

---

# 5. Machine Learning Models

## 5.1 LightGBM / XGBoost

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Gradient boosted trees on hand-crafted lag/calendar/rolling features |
| **Regime handling** | Implicit via tree splits — must engineer regime-aware features |
| **Best for** | Many series (global model), many features, nonlinear patterns |
| **Avoid when** | Very short series, need native probabilistic output, no features |
| **Uncertainty** | Quantile regression or conformal prediction add-on |
| **Python** | `lightgbm`, `xgboost` |
| **Status** | *To be expanded* |

---

## 5.2 Random Forest

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Ensemble of decision trees on lag/calendar features |
| **Best for** | Similar to LightGBM but less tuning; good baseline |
| **Avoid when** | Same as LightGBM; generally outperformed by boosting methods |
| **Python** | `scikit-learn` |
| **Status** | *To be expanded* |

---

## 5.3 Linear Models

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Ridge/Lasso/ElasticNet on lag features — fast, regularized, interpretable |
| **Best for** | Many series with similar behavior, feature selection (Lasso) |
| **Avoid when** | Nonlinear dynamics |
| **Python** | `scikit-learn` |
| **Status** | *To be expanded* |

---

# 6. Deep Learning / Neural Models

## 6.1 DeepAR (Amazon)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Autoregressive RNN that outputs probability distributions at each timestep |
| **Regime handling** | Implicit — LSTM hidden state adapts to regime context |
| **Best for** | Many related series, probabilistic forecasts, cold-start via cross-learning |
| **Avoid when** | Few series, need interpretability, limited compute |
| **Python** | `gluonts`, `pytorch-forecasting` |
| **Status** | *To be expanded* |

---

## 6.2 N-BEATS / N-HiTS

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Pure DL architecture with interpretable stacked blocks; N-HiTS adds hierarchical interpolation |
| **Best for** | Strong univariate forecasting, no covariates needed |
| **Avoid when** | Need exogenous variables (original N-BEATS), small datasets |
| **Python** | `neuralforecast`, `pytorch-forecasting` |
| **Status** | *To be expanded* |

---

## 6.3 Temporal Fusion Transformer (TFT)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Attention-based model with variable selection, static/temporal covariates, interpretable |
| **Best for** | Complex forecasting with many covariates, interpretability among DL models |
| **Key feature** | Attention weights show WHICH time steps and features the model focuses on |
| **Avoid when** | Small datasets, no covariates (overkill) |
| **Python** | `pytorch-forecasting` |
| **Status** | *To be expanded* |

---

## 6.4 WaveNet

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Dilated causal convolutions — originally for audio, adapted for time series |
| **Best for** | Very long sequences, audio-like signals |
| **Avoid when** | Standard forecasting tasks (usually outperformed by TFT/N-BEATS) |
| **Status** | *To be expanded* |

---

## 6.5 Informer / Autoformer / PatchTST

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Transformer variants optimized for long-horizon time series forecasting |
| **Informer** | ProbSparse attention for efficiency |
| **Autoformer** | Auto-correlation mechanism replaces attention |
| **PatchTST** | Patches input into tokens like ViT; channel-independent |
| **Best for** | Long-horizon forecasting, multivariate |
| **Python** | `neuralforecast`, HuggingFace |
| **Status** | *To be expanded* |

---

# 7. Foundation Models

## 7.1 Chronos (Amazon)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Pre-trained on millions of time series; tokenizes values and uses language model architecture |
| **Best for** | Zero-shot forecasting, quick baselines without training |
| **Avoid when** | Domain-specific series far from training distribution |
| **Python** | `chronos-forecasting` |
| **Status** | *To be expanded* |

---

## 7.2 TimesFM (Google)

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Foundation model for time series; patched decoder-only transformer |
| **Best for** | Zero-shot / few-shot forecasting |
| **Python** | `timesfm` |
| **Status** | *To be expanded* |

---

## 7.3 Moirai / Lag-Llama

| Attribute | Detail |
|-----------|--------|
| **Moirai** | Universal forecasting transformer (Salesforce) |
| **Lag-Llama** | Foundation model using lags as tokens |
| **Best for** | Zero-shot probabilistic forecasting |
| **Status** | *To be expanded* |

---

# 8. Specialized Methods

## 8.1 Croston / SBA / TSB

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Models for intermittent demand (many zeros with occasional non-zero spikes) |
| **Croston** | Separately smooths demand size and inter-arrival intervals |
| **SBA** | Bias-corrected Croston |
| **TSB** | Handles obsolescence (demand trending to zero) |
| **Best for** | Spare parts, slow-moving inventory, intermittent demand |
| **Python** | `statsforecast` |
| **Status** | *To be expanded* |

---

## 8.2 Hierarchical Reconciliation

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Ensures forecasts at different aggregation levels are coherent |
| **Methods** | Bottom-up, Top-down, MinT (optimal combination) |
| **Best for** | Retail (store → region → national), product hierarchies |
| **Python** | `hierarchicalforecast` |
| **Status** | *To be expanded* |

---

## 8.3 FFORMA

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Feature-based Forecast Model Averaging — uses series features to weight models |
| **Best for** | Automated model selection across many diverse series |
| **Status** | *To be expanded* |

---

## 8.4 Ensemble / Stacking

| Attribute | Detail |
|-----------|--------|
| **One-liner** | Combine forecasts from multiple models (simple average, weighted, or meta-learner) |
| **Best for** | Almost always improves over single model; competition-winning strategy |
| **Key insight** | Simple average is a surprisingly strong baseline |
| **Status** | *To be expanded* |

---

# Comparison by Phenomenon

> [!IMPORTANT]
> This section is the quick-lookup for "I have X, which model should I use?"

## By Data Characteristics

| Phenomenon | First Choice | Strong Alternative | Avoid |
|------------|-------------|-------------------|-------|
| **Trend + single seasonality** | SARIMA | ETS (Holt-Winters) | Plain ARIMA |
| **Multiple seasonalities** | TBATS, MSTL+ARIMA | Prophet | SARIMA (one period only) |
| **Regime changes (recurring)** | Markov-Switching | HMM, TAR/SETAR | ARIMA (assumes stability) |
| **Regime changes (permanent)** | Bai-Perron, PELT | Prophet (changepoints) | Markov-Switching |
| **Volatility clustering** | GARCH family | EGARCH (asymmetric) | ARIMA (constant variance) |
| **Long memory** | ARFIMA | HAR (Heterogeneous AR) | ARIMA (limited lags) |
| **Intermittent demand** | Croston, SBA, TSB | LightGBM with zero-inflation features | ARIMA, ETS |
| **Many exogenous variables** | LightGBM, XGBoost | TFT | ARIMA (linear only) |
| **Few known regressors** | ARIMAX, Prophet | BSTS | Pure ML (overkill) |
| **Multivariate (causal links)** | VAR, VECM | Granger causality + ARIMAX | Independent ARIMAs |
| **1000+ series** | LightGBM (global), DeepAR | TFT, N-HiTS | Per-series ARIMA (slow) |
| **Cold start / new product** | DeepAR (cross-learning) | Similar-item transfer | Any single-series model |
| **Short series (< 30 obs)** | ETS, Theta | Simple average/naive | Complex DL models |
| **Need interpretability** | ARIMA, ETS, Prophet, BSTS | LightGBM+SHAP, TFT | DeepAR, N-BEATS |
| **Need uncertainty** | DeepAR, BSTS | Conformal on any model | Point-only models |

## By Industry / Domain

| Domain | Common Models | Why |
|--------|--------------|-----|
| **Retail / demand** | LightGBM, DeepAR, Prophet, Croston | Many series, intermittent, promotions |
| **Finance / risk** | GARCH, Markov-Switching, ARIMA+GARCH | Volatility modeling, regime shifts |
| **Supply chain** | LightGBM, ARIMAX, hierarchical | Covariates, hierarchy, scale |
| **Energy / utilities** | TBATS, Prophet, TFT | Multiple seasonalities, weather covariates |
| **Macro-economics** | VAR, VECM, BSTS | Multivariate, policy impact, Bayesian |
| **Web traffic** | Prophet, LightGBM | Holidays, trend changes, scale |
| **IoT / sensor** | N-BEATS, DeepAR, BOCPD | High frequency, anomaly detection |

---

*Parent: [Time Series README](./README.md) | [Study Plan](../time_series_study_plan.md) | [Main Agenda](../as_study_agenda.md)*
