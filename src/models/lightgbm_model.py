"""
LightGBM cross-validation script for mold features.

This script fits LightGBM models per fold and writes
out a CSV per fold with: [name, label, pred, logit_0, logit_1, ...]

Usage:
    python lightgbm_model.py \
        --features features.npz \
        --folds_dir molddata/folds \
        --output_dir results/lgbm_cv \
        [--num_class 5] [--num_leaves 31] [--learning_rate 0.005] [--feature_fraction 0.9]
"""


import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_features(feature_path: str):
    """Load npz features/labels as arrays."""
    data = np.load(feature_path)
    return data['color'], data['lbp'], data['shape'], data['y']


def get_fold_files(folds_dir: str):
    """Get sorted train/val CSV lists for cross-validation."""
    train_files = sorted(
        os.path.join(folds_dir, f)
        for f in os.listdir(folds_dir)
        if f.startswith('train_fold_') and f.endswith('.csv')
    )
    val_files = sorted(
        os.path.join(folds_dir, f)
        for f in os.listdir(folds_dir)
        if f.startswith('val_fold_') and f.endswith('.csv')
    )
    return train_files, val_files


def standardize_features(color, lbp, shape):
    """Standardize per feature type and concatenate features."""
    scaler_color = StandardScaler().fit(color)
    scaler_lbp = StandardScaler().fit(lbp)
    scaler_shape = StandardScaler().fit(shape)
    color_std = scaler_color.transform(color)
    lbp_std = scaler_lbp.transform(lbp)
    shape_std = scaler_shape.transform(shape)
    X_std = np.hstack([color_std, lbp_std, shape_std])
    return X_std, scaler_color, scaler_lbp, scaler_shape


def fit_and_predict_lgbm(
    X_train, y_train, X_val, num_class=5, num_leaves=31, learning_rate=0.005, feature_fraction=0.9
):
    """Train a LightGBM model and return logits for validation databases."""
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain)
    logits = model.predict(X_val)  # shape [N_val, num_class]
    preds = np.argmax(logits, axis=1)
    return preds, logits, model


def save_preds_logits(names, labels, preds, logits, output_file):
    """Save sample names, gold labels, predicted labels, and logits as a CSV."""
    df = pd.DataFrame({
        'name': names,
        'label': labels,
        'pred': preds
    })
    for i in range(logits.shape[1]):
        df[f'logit_{i}'] = logits[:, i]
    df.to_csv(output_file, index=False)