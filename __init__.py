import base64
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Blueprint, jsonify, render_template, request
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_META = {
    'id': 'work-life-regression',
    'name': 'Longevity Prediction',
    'description': 'Predict lifespan from lifestyle habits using four regression models with GridSearchCV tuning, plus EDA with correlation heatmap, stacked histogram, and dot plot.',
    'icon': 'trending_up',
    'color': '#7c3aed',
    'category': 'Supervised Regression',
    'nav_group': 'Machine Learning',
    'tags': ['regression', 'gridsearchcv', 'random forest', 'gradient boosting', 'ridge', 'lasso'],
}

bp = Blueprint(
    'work_life_regression',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='static',
    url_prefix='/work-life-regression',
)

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'Updated Quality of Life Data.csv',
)

# Per-model definition: estimator + param_grid
_MODELS = {
    'random_forest': {
        'estimator': lambda: RandomForestRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
        },
    },
    'gradient_boosting': {
        'estimator': lambda: GradientBoostingRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        },
    },
    'ridge': {
        'estimator': lambda: Ridge(),
        'param_grid': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    },
    'lasso': {
        'estimator': lambda: Lasso(max_iter=5000),
        'param_grid': {'alpha': [0.01, 0.1, 1.0, 10.0]},
    },
}

_cache = {}               # keyed by model name
_diagnostics_cache = {}  # learning curve, keyed by model name
_preprocessed = None      # shared train/test splits (same for all models)
_plots_cache = None       # cached base64 plot images

# ── Shared dark-theme palette ─────────────────────────────────────
_BG      = '#1e293b'
_SURFACE = '#334155'
_TEXT    = '#f1f5f9'
_MUTED   = '#94a3b8'
_ACCENT  = '#7c3aed'
_CYAN    = '#00d4ff'


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def _load_and_split():
    global _preprocessed
    if _preprocessed is not None:
        return _preprocessed

    df = pd.read_csv(_DATA_PATH)
    df = df.drop(columns=['id'])

    X = df.drop(columns=['age_at_death'])
    y = df['age_at_death']

    X = pd.get_dummies(X, columns=['gender', 'occupation_type'], drop_first=False)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    _preprocessed = {
        'X_train': X_train_scaled,
        'X_test':  X_test_scaled,
        'y_train': y_train,
        'y_test':  y_test,
        'feature_names': feature_names,
        'n_train': len(X_train),
        'n_test':  len(X_test),
    }
    return _preprocessed


@bp.route('/')
def index():
    return render_template('work_life_regression/index.html')


@bp.route('/plots')
def plots():
    global _plots_cache
    if _plots_cache is not None:
        return jsonify(_plots_cache)

    df = pd.read_csv(_DATA_PATH).drop(columns=['id'])
    num_cols = ['avg_work_hours_per_day', 'avg_rest_hours_per_day',
                'avg_sleep_hours_per_day', 'avg_exercise_hours_per_day', 'age_at_death']

    BG      = '#1e293b'
    SURFACE = '#334155'
    TEXT    = '#f1f5f9'
    MUTED   = '#94a3b8'
    ACCENT  = '#7c3aed'
    PALETTE = ['#7c3aed', '#00d4ff', '#f472b6', '#fbbf24', '#10b981',
               '#f97316', '#ec4899', '#14b8a6']

    sns.set_theme(style='dark', rc={
        'figure.facecolor': BG, 'axes.facecolor': BG,
        'axes.edgecolor': SURFACE, 'axes.labelcolor': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'text.color': TEXT, 'grid.color': SURFACE,
        'font.family': 'sans-serif',
    })

    # ── 1. Correlation heatmap ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax,
                cmap=sns.diverging_palette(260, 330, s=80, l=45, as_cmap=True),
                linewidths=0.5, linecolor=BG, annot_kws={'size': 9},
                cbar_kws={'shrink': 0.8})
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    heatmap_b64 = _fig_to_b64(fig)

    # ── 2. Stacked histogram ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    occupations = sorted(df['occupation_type'].unique())
    colors = sns.color_palette('husl', len(occupations))
    bins = range(int(df['age_at_death'].min()), int(df['age_at_death'].max()) + 2)
    data_by_occ = [df[df['occupation_type'] == occ]['age_at_death'].values
                   for occ in occupations]
    ax.hist(data_by_occ, bins=bins, stacked=True, color=colors,
            alpha=0.85, label=occupations, edgecolor='none')
    ax.legend(fontsize=7.5, framealpha=0.15, labelcolor=TEXT,
              facecolor=SURFACE, edgecolor='none')
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    histogram_b64 = _fig_to_b64(fig)

    # ── 3. PairGrid dot plot (seaborn example style) ──────────────
    x_vars = ['avg_work_hours_per_day', 'avg_rest_hours_per_day',
              'avg_sleep_hours_per_day', 'avg_exercise_hours_per_day',
              'age_at_death']
    col_titles = ['Work Hours', 'Rest Hours', 'Sleep Hours',
                  'Exercise Hours', 'Age at Death']

    agg = (df.groupby('occupation_type')[x_vars]
             .mean()
             .reset_index()
             .sort_values('age_at_death', ascending=False))

    sns.set_theme(style='whitegrid', rc={
        'figure.facecolor': BG, 'axes.facecolor': BG,
        'axes.edgecolor': SURFACE, 'axes.labelcolor': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'text.color': TEXT, 'grid.color': SURFACE,
        'font.family': 'sans-serif',
    })

    g = sns.PairGrid(agg, x_vars=x_vars, y_vars=['occupation_type'],
                     height=6, aspect=0.4)

    g.map(sns.stripplot, size=10, orient='h', jitter=False,
          palette='flare_r', linewidth=1, edgecolor=BG)

    g.set(ylabel='')

    for ax, title in zip(g.axes.flat, col_titles):
        ax.set_title(title, color=TEXT, fontsize=9, pad=8)
        ax.set_xlabel('')
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=8, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    g.figure.set_facecolor(BG)
    sns.despine(left=True, bottom=True)
    g.figure.tight_layout()
    pairgrid_b64 = _fig_to_b64(g.figure)

    _plots_cache = {
        'heatmap':   heatmap_b64,
        'histogram': histogram_b64,
        'pairgrid':  pairgrid_b64,
    }
    return jsonify(_plots_cache)


@bp.route('/run')
def run():
    model_key = request.args.get('model', 'random_forest')
    if model_key not in _MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400

    force = request.args.get('force', 'false').lower() == 'true'
    if model_key in _cache and not force:
        return jsonify(_cache[model_key])

    data = _load_and_split()
    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_test  = data['y_test']
    feature_names = data['feature_names']

    cfg = _MODELS[model_key]
    grid_search = GridSearchCV(
        cfg['estimator'](),
        cfg['param_grid'],
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    y_pred       = best.predict(X_test)
    y_train_pred = best.predict(X_train)

    # Metrics
    test_r2   = float(r2_score(y_test, y_pred))
    test_mae  = float(mean_absolute_error(y_test, y_pred))
    test_mse  = float(mean_squared_error(y_test, y_pred))
    test_rmse = float(np.sqrt(test_mse))

    train_r2   = float(r2_score(y_train, y_train_pred))
    train_mae  = float(mean_absolute_error(y_train, y_train_pred))
    train_mse  = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))

    # Feature importance (tree models) or normalised |coef_| (linear models)
    if hasattr(best, 'feature_importances_'):
        raw = best.feature_importances_
    else:
        raw = np.abs(best.coef_)
        total = raw.sum()
        raw = raw / total if total > 0 else raw

    feat_imp = sorted(
        [{'feature': f, 'importance': float(v)} for f, v in zip(feature_names, raw)],
        key=lambda x: x['importance'],
        reverse=True,
    )

    # Sample for scatter
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_pred), min(500, len(y_pred)), replace=False)
    y_test_arr = np.array(y_test)

    best_params = {
        k: (v if v is not None else 'None')
        for k, v in grid_search.best_params_.items()
    }

    # ── Residuals plots ───────────────────────────────────────────
    residuals = np.array(y_test) - y_pred

    sns.set_theme(style='dark', rc={
        'figure.facecolor': _BG, 'axes.facecolor': _BG,
        'axes.edgecolor': _SURFACE, 'axes.labelcolor': _TEXT,
        'xtick.color': _MUTED, 'ytick.color': _MUTED,
        'text.color': _TEXT, 'grid.color': _SURFACE,
        'font.family': 'sans-serif',
    })

    # Residuals vs Fitted
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=12,
               color=_ACCENT, edgecolors='none')
    ax.axhline(0, color=_CYAN, linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_xlabel('Fitted Values (Predicted Age)', color=_TEXT)
    ax.set_ylabel('Residual (Actual − Predicted)', color=_TEXT)
    ax.set_title('Residuals vs Fitted', color=_TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_SURFACE)
    fig.tight_layout()
    residuals_plot_b64 = _fig_to_b64(fig)

    # Residuals distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, color=_ACCENT, alpha=0.75, edgecolor='none')
    ax.axvline(0, color=_CYAN, linewidth=1.5, linestyle='--', alpha=0.8, label='Zero')
    ax.axvline(residuals.mean(), color=_MUTED, linewidth=1.2,
               linestyle=':', alpha=0.8, label=f'Mean {residuals.mean():.2f}')
    ax.set_xlabel('Residual (years)', color=_TEXT)
    ax.set_ylabel('Count', color=_TEXT)
    ax.set_title('Residuals Distribution', color=_TEXT, fontsize=12, pad=10)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=_TEXT,
              facecolor=_SURFACE, edgecolor='none')
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_SURFACE)
    fig.tight_layout()
    residuals_dist_b64 = _fig_to_b64(fig)

    _cache[model_key] = {
        'metrics': {
            'test_r2':    round(test_r2,   4),
            'test_mae':   round(test_mae,  4),
            'test_mse':   round(test_mse,  4),
            'test_rmse':  round(test_rmse, 4),
            'train_r2':   round(train_r2,   4),
            'train_mae':  round(train_mae,  4),
            'train_mse':  round(train_mse,  4),
            'train_rmse': round(train_rmse, 4),
            'cv_score':   round(float(grid_search.best_score_), 4),
        },
        'best_params': best_params,
        'actual':    [float(y_test_arr[i]) for i in idx],
        'predicted': [float(y_pred[i]) for i in idx],
        'feature_importance': feat_imp,
        'n_train': data['n_train'],
        'n_test':  data['n_test'],
        'model_key': model_key,
        'residuals_plot': residuals_plot_b64,
        'residuals_dist': residuals_dist_b64,
    }
    return jsonify(_cache[model_key])


@bp.route('/diagnostics')
def diagnostics():
    model_key = request.args.get('model', 'random_forest')
    if model_key not in _MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 400

    if model_key in _diagnostics_cache:
        return jsonify(_diagnostics_cache[model_key])

    data    = _load_and_split()
    X_train = data['X_train']
    y_train = data['y_train']

    # Clone the best estimator if already trained, otherwise use default
    if model_key in _cache:
        # Re-instantiate with best params to avoid fitting issues
        cfg = _MODELS[model_key]
        best_params = _cache[model_key]['best_params']
        estimator = cfg['estimator']()
        estimator.set_params(**{k: v for k, v in best_params.items()
                                if v != 'None'})
    else:
        estimator = _MODELS[model_key]['estimator']()

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=3, scoring='r2', n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    sns.set_theme(style='dark', rc={
        'figure.facecolor': _BG, 'axes.facecolor': _BG,
        'axes.edgecolor': _SURFACE, 'axes.labelcolor': _TEXT,
        'xtick.color': _MUTED, 'ytick.color': _MUTED,
        'text.color': _TEXT, 'grid.color': _SURFACE,
        'font.family': 'sans-serif',
    })

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(train_sizes, train_mean, 'o-', color=_ACCENT, linewidth=2.2,
            markersize=5, label='Training R²')
    ax.fill_between(train_sizes,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.12, color=_ACCENT)

    ax.plot(train_sizes, val_mean, 'o-', color=_CYAN, linewidth=2.2,
            markersize=5, label='Cross-val R²')
    ax.fill_between(train_sizes,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.12, color=_CYAN)

    ax.set_xlabel('Training Set Size', color=_TEXT)
    ax.set_ylabel('R² Score', color=_TEXT)
    ax.set_title('Learning Curve', color=_TEXT, fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.15, labelcolor=_TEXT,
              facecolor=_SURFACE, edgecolor='none')
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_SURFACE)
    fig.tight_layout()
    lc_b64 = _fig_to_b64(fig)

    _diagnostics_cache[model_key] = {'learning_curve': lc_b64, 'model_key': model_key}
    return jsonify(_diagnostics_cache[model_key])
