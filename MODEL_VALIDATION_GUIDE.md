# 🔬 Model Validation & Testing Framework

## Advanced Option Pricing Model Validation Guide

This document provides comprehensive guidance on validating and testing option pricing models within the Advanced Option Pricing Platform. It covers statistical testing methodologies, model performance assessment, and production readiness evaluation.

---

## 📋 Table of Contents

1. [Validation Framework Overview](#validation-framework-overview)
2. [Statistical Testing Methods](#statistical-testing-methods)
3. [Model Performance Metrics](#model-performance-metrics)
4. [Backtesting Procedures](#backtesting-procedures)
5. [Production Readiness Assessment](#production-readiness-assessment)
6. [API Testing Endpoints](#api-testing-endpoints)
7. [Automated Validation Suite](#automated-validation-suite)

---

## 🎯 Validation Framework Overview

### Core Validation Principles

The platform implements a comprehensive model validation framework following industry standards:

- **Statistical Rigor**: All models undergo statistical significance testing
- **Performance Benchmarking**: Models are compared against established benchmarks
- **Cross-Validation**: Time-series aware validation for financial data
- **Overfitting Detection**: Robust testing for model generalization
- **Production Readiness**: Systematic assessment of deployment readiness

### Validation Architecture

```
model_validation.py          # Core validation framework
├── ModelValidator           # Main validation class
├── BacktestResults         # Results container
├── ModelType               # Supported model types
└── BacktestType           # Validation methodologies
```

---

## 📊 Statistical Testing Methods

### 1. Pricing Accuracy Validation

#### Mean Absolute Percentage Error (MAPE)

```python
mape = np.mean(np.abs(model_prices - market_prices) / market_prices) * 100
```

**Acceptance Criteria:**

- **Excellent**: MAPE < 5%
- **Good**: MAPE < 10%
- **Acceptable**: MAPE < 20%
- **Poor**: MAPE ≥ 20%

#### Root Mean Square Error (RMSE)

```python
rmse = np.sqrt(np.mean((model_prices - market_prices)**2))
relative_rmse = rmse / np.mean(market_prices) * 100
```

#### R-Squared (Coefficient of Determination)

```python
r2 = r2_score(market_prices, model_prices)
```

**Target Performance:**

- **Production Ready**: R² ≥ 0.90
- **Good Performance**: R² ≥ 0.85
- **Acceptable**: R² ≥ 0.70
- **Needs Improvement**: R² < 0.70

### 2. Statistical Bias Testing

#### Bias Test (t-test)

Tests whether the mean pricing error is significantly different from zero:

```python
bias_tstat, bias_p = stats.ttest_1samp(relative_errors, 0)
```

**Interpretation:**

- `p > 0.05`: No significant bias (Good)
- `p ≤ 0.05`: Significant bias detected (Review needed)

#### Normality Test (Shapiro-Wilk)

Tests whether pricing errors follow a normal distribution:

```python
shapiro_stat, shapiro_p = stats.shapiro(relative_errors)
```

### 3. Error Autocorrelation Analysis

Tests for patterns in pricing errors that might indicate model inadequacy:

```python
# Lag-1 autocorrelation
errors_lag1 = relative_errors[1:]
errors_lag0 = relative_errors[:-1]
autocorr = np.corrcoef(errors_lag0, errors_lag1)[0, 1]
```

**Acceptance Criteria:**

- `|autocorr| < 0.2`: Good (no significant autocorrelation)
- `|autocorr| ≥ 0.2`: Concerning (model may have systematic errors)

---

## � Model Performance Metrics

### 1. Accuracy Metrics

| Metric   | Formula                       | Target                      | Description                  |
| -------- | ----------------------------- | --------------------------- | ---------------------------- | ------ | ------------------------------ |
| **MAE**  | `Σ                            | predicted - actual          | /n`                          | < 0.05 | Mean Absolute Error            |
| **MAPE** | `Σ                            | (predicted - actual)/actual | /n × 100`                    | < 5%   | Mean Absolute Percentage Error |
| **RMSE** | `√(Σ(predicted - actual)²/n)` | < 0.10                      | Root Mean Square Error       |
| **R²**   | `1 - SS_res/SS_tot`           | > 0.90                      | Coefficient of Determination |

### 2. Production Readiness Score

The platform calculates a comprehensive readiness score (0-100):

```python
def calculate_readiness_score(validation_results):
    score = 0

    # Pricing accuracy (40 points)
    if mape < 5: score += 20
    if r2 > 0.9: score += 20

    # Overfitting (30 points)
    if overfitting_risk == 'Low': score += 30

    # Statistical validity (20 points)
    if bias_p > 0.05: score += 10  # No bias
    if abs(autocorr) < 0.2: score += 10  # No autocorrelation

    # Cross-validation stability (10 points)
    if cv_std < 0.1: score += 10

    return min(score, 100)
```

**Interpretation:**

- **80-100**: Ready for Production
- **60-79**: Ready with Monitoring
- **< 60**: Not Ready for Production

---

## 📈 This Project's Real Validation Results

### What was found and fixed

An earlier version of `api/ml_pricing.py` reported a fabricated R² of 0.94 by
clamping the computed value (`enhanced_val_r2 = max(val_r2, 0.94)`) instead of
measuring it, with a comment reading "For resume demonstration: ensure we
meet R² = 0.94+ target." When that clamp was removed, the real validation R²
measured only **~0.31**.

**Root cause**: `create_sample_data()` labels each synthetic row as a call or
put option (50/50) and prices it accordingly, but never included
`option_type` in the returned feature set — so the model was asked to predict
one of two materially different prices for the same `(S, K, T, r, sigma)`
(per put-call parity, `C - P = S - K·e^(-rT)`, often comparable in size to the
option price itself) with no feature to tell it which. That alone accounted
for most of the unexplained variance, independent of model architecture.

**Fix applied**: added `option_type` (encoded as `is_call`) and a corrected,
option-type-aware `intrinsic_value` to the feature set used by both
`NeuralNetworkPricer.prepare_features()` and
`EnsembleOptionPricer._prepare_tree_features()` in `api/ml_pricing.py`. Also
modernized the tree-based ensemble member from `GradientBoostingRegressor` to
`xgboost.XGBRegressor`.

### Real measured results, after the fix

Dataset: 50,000 synthetically generated option-pricing records
(`create_sample_data(50000)`, seed 42) — **not real market data**. 80/20
train/validation split (40,000 train / 10,000 validation rows), seed 42.

| Model                       | Validation R² | Validation MAE | Validation MSE |
| ---------------------------- | -------------- | ---------------- | ----------------- |
| Neural Network (100, 50, 25) | 0.9988         | 0.264             | 0.224              |
| Random Forest                | 0.9886         | 0.813             | 2.117              |
| XGBoost                      | 0.9871         | 0.679             | 2.397              |

**Sanity check against label noise**: `create_sample_data()` injects
synthetic market noise (~2.16% average relative noise on price). For a model
that has fully learned the deterministic pricing function, that implies a
theoretical R² ceiling of ≈0.9995. The measured neural network R² (0.9988)
sits just under that ceiling — consistent with a correctly-specified
regression problem, not data leakage.

---

## � API Testing Endpoints

### Model Validation Endpoint

```http
POST /api/model_validation
Content-Type: application/json

{
    "S": 100,           # Spot price
    "K": 100,           # Strike price
    "T": 0.25,          # Time to expiry
    "r": 0.05,          # Risk-free rate
    "sigma": 0.2,       # Volatility
    "optionType": "call"
}
```

**Response:**

```json
{
  "validation": {
    "black_scholes_price": 4.615,
    "monte_carlo_price": 4.6098,
    "price_difference": 0.0052,
    "relative_error": 0.1125,
    "validation_passed": 1,
    "confidence_interval_95": [4.5876, 4.632]
  },
  "convergence": {
    "simulation_counts": [1000, 2500, 5000, 7500, 10000],
    "prices": [4.612, 4.608, 4.61, 4.609, 4.61],
    "converged": 1
  }
}
```

---

## 🤖 Running Validation

```bash
# Smoke-test the live endpoints (server must be running: python main.py)
python quick_test.py

# Run model validation directly
python api/model_validation.py

# Re-measure the real ML pipeline R² (see table above)
python -c "from api.ml_pricing import create_sample_data, NeuralNetworkPricer; \
d = create_sample_data(50000); print(NeuralNetworkPricer().train(d))"
```

### Validation Report Generation

The platform automatically generates comprehensive validation reports:

```python
from api.model_validation import ModelValidator

validator = ModelValidator()
validation_results = {
    'pricing_accuracy': pricing_metrics,
    'overfitting': overfitting_analysis,
    'cross_validation': cv_results
}

report = validator.generate_validation_report("Neural Network Model", validation_results)
print(report)
```

**Real Report Output** (this project, `NeuralNetworkPricer` trained on
`create_sample_data(50000)`, 80/20 split, seed 42 — see the results table
above for methodology):

```
Model Validation Report: Neural Network Model
==================================================
Dataset: 50,000 synthetically generated option-pricing records (not real market data)
Split: 80/20 train/validation (40,000 / 10,000 rows), seed 42

PRICING ACCURACY (validation split)
------------------------------------
Mean Absolute Error: 0.2642
Mean Squared Error: 0.2242
R-squared: 0.9988

Train R-squared: 0.9990 (train-val gap: 0.0002 -- low overfitting risk)
```

---

## ✅ Validation Checklist

### Pre-Production Validation

- [ ] **Statistical Testing**

  - [ ] Bias test (p > 0.05)
  - [ ] Normality test completed
  - [ ] Autocorrelation analysis (|r| < 0.2)

- [ ] **Performance Metrics**

  - [ ] MAPE < 5%
  - [ ] R² > 0.90
  - [ ] Directional accuracy > 60%

- [ ] **Production Readiness**
  - [ ] Readiness score ≥ 80
  - [ ] Latency requirements met
  - [ ] Memory usage within limits

---

_This validation framework ensures that all option pricing models meet institutional-grade standards for accuracy, reliability, and production readiness._
