"""
LightGBM cross-validation script for mold radiomics features.

Usage:
    python lgbm.py \
        --features features.npz \
        --folds_dir molddata/folds \
        --output_dir results/lgbm_cv \
        [--num_class 5] [--num_leaves 31] [--learning_rate 0.005] [--feature_fraction 0.9]
"""
import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt


def load_features(feature_path: str):
    """
    Load radiomic features and labels from a .npz file.

    Returns:
        color: np.ndarray of shape (N, C1)
        lbp:   np.ndarray of shape (N, C2)
        shape: np.ndarray of shape (N, C3)
        y:     np.ndarray of labels, shape (N,)
    """
    data = np.load(feature_path)
    return data['color'], data['lbp'], data['shape'], data['y']


def get_fold_files(folds_dir: str):
    """
    Collect sorted lists of train/val CSV files for cross-validation.
    """
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


def cross_validate(
    color: np.ndarray,
    lbp: np.ndarray,
    shape: np.ndarray,
    y: np.ndarray,
    train_files: list,
    val_files: list,
    params: dict,
    output_dir: str
):
    """
    Perform k-fold cross-validation with LightGBM, saving metrics and plots.
    """
    # Standardize features
    scaler_color = StandardScaler().fit(color)
    scaler_lbp   = StandardScaler().fit(lbp)
    scaler_shape = StandardScaler().fit(shape)

    color_std = scaler_color.transform(color)
    lbp_std   = scaler_lbp.transform(lbp)
    shape_std = scaler_shape.transform(shape)

    n_folds = len(train_files)
    metrics = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'specificity': [],
    }

    for fold_idx, (train_csv, val_csv) in enumerate(zip(train_files, val_files)):
        # Load indices
        train_idx = pd.read_csv(train_csv, index_col=0).index.values
        val_idx   = pd.read_csv(val_csv,   index_col=0).index.values

        # Prepare training and validation sets
        X_train = np.hstack((
            color_std[train_idx],
            lbp_std[train_idx],
            shape_std[train_idx]
        ))
        y_train = y[train_idx]

        X_val = np.hstack((
            color_std[val_idx],
            lbp_std[val_idx],
            shape_std[val_idx]
        ))
        y_val = y[val_idx]

        # Train LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain)

        # Predict probabilities and classes
        y_prob = model.predict(X_val)
        y_pred = np.argmax(y_prob, axis=1)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        prec = precision_score(y_val, y_pred, average='macro')
        rec = recall_score(y_val, y_pred, average='macro')
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['specificity'].append(spec)

        # Save feature importance plot
        plt.figure()
        lgb.plot_importance(model, max_num_features=10)
        plt.title(f'Feature Importance Fold {fold_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_fold_{fold_idx}.png'))
        plt.close()

    # Aggregate and save metrics
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, 'cv_metrics.csv'), index=False)

    print("Cross-validation results:")
    print(df.describe().loc[['mean', 'std']])


def main():
    parser = argparse.ArgumentParser(description="LightGBM 5-fold CV for mold radiomics features.")
    parser.add_argument('--features',   type=str, required=True,
                        help='Path to .npz file with radiomics features')
    parser.add_argument('--folds_dir',  type=str, required=True,
                        help='Directory containing train_fold_*.csv and val_fold_*.csv')
    parser.add_argument('--output_dir', type=str, default='results/lgbm_cv',
                        help='Directory to save results')
    parser.add_argument('--num_class',      type=int,   default=5)
    parser.add_argument('--num_leaves',     type=int,   default=31)
    parser.add_argument('--learning_rate',  type=float, default=0.005)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    color, lbp, shape, y = load_features(args.features)

    # Encode labels if non-numeric
    if y.dtype.kind not in 'biufc':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Prepare CV file lists
    train_files, val_files = get_fold_files(args.folds_dir)

    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': args.num_class,
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': args.feature_fraction,
    }

    # Run CV
    cross_validate(color, lbp, shape, y, train_files, val_files, params, args.output_dir)


if __name__ == '__main__':
    main()
