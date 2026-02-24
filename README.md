# Longevity Prediction

Predict lifespan from lifestyle habits using four regression models with GridSearchCV hyperparameter tuning, plus EDA and diagnostic visualizations.

## Overview

This project builds a regression pipeline to predict age at death from features like work hours, sleep, exercise, and occupation type. GridSearchCV tunes each model's hyperparameters via 3-fold cross-validation. All training runs on the backend; the frontend renders metrics and plots without a page reload.

## Models

| Model | GridSearchCV parameters |
|-------|------------------------|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| Ridge Regression | `alpha` (0.1, 1.0, 10.0, 100.0) |
| Lasso Regression | `alpha` (0.01, 0.1, 1.0, 10.0) |

## Features

- **EDA plots** — correlation heatmap, stacked histogram by occupation, PairGrid dot plot
- **Metrics** — test/train R², MAE, MSE, RMSE, cross-validation R², best hyperparameters
- **Diagnostic plots** — prediction scatter, feature importance, residuals vs fitted, residuals distribution, learning curve
- **Caching** — preprocessed data and trained models are cached in memory so re-running is instant

## Dataset

"Quality of Life" dataset with lifestyle and demographic features:

| Feature | Type |
|---------|------|
| `work_hours` | Numeric |
| `rest_hours` | Numeric |
| `sleep_hours` | Numeric |
| `exercise_hours` | Numeric |
| `occupation_type` | Categorical (one-hot encoded) |
| `gender` | Categorical (one-hot encoded) |
| **`age_at_death`** | **Target** |

Train/test split: 80/20. Numeric features scaled with StandardScaler.

## Tech Stack

- **Backend:** Flask, scikit-learn, pandas, numpy, matplotlib, seaborn
- **Frontend:** Chart.js 3.9.1, Vanilla JavaScript

## Project Structure

```
work_life_regression/
├── __init__.py                       # Flask blueprint, training pipeline, GridSearchCV, API routes
├── templates/
│   └── work_life_regression/
│       └── index.html                # Model selector, metrics grid, EDA + diagnostic plot panels
└── static/
    └── script.js                     # Fetch calls, Chart.js feature importance bar chart
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/work-life-regression/` | Main page |
| GET | `/work-life-regression/run?model=<key>&force=<bool>` | Train a model, return metrics + plots |
| GET | `/work-life-regression/plots` | Generate 3 EDA plots as base64 PNG |
| GET | `/work-life-regression/diagnostics?model=<key>` | Generate learning curve for a model |

**Model keys:** `random_forest`, `gradient_boosting`, `ridge`, `lasso`
