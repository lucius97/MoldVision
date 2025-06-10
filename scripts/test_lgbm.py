import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from configs.config import load_yaml_file
from src.models.lightgbm_model import load_features, get_fold_files, standardize_features, save_preds_logits

def main():
    parser = argparse.ArgumentParser(description="Test LightGBM model for mold classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--features_path", type=str, help="Path to features.npz file")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index to test")
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_file(args.config)

    # Set up directories
    data_dir = config.get("DATA_DIR", "data")
    features_dir = config.get("FEATURES_DIR", "features")
    output_dir = config.get("OUTPUT_DIR", "output")
    folds_dir = os.path.join(data_dir, "folds")

    # Use provided features path or default
    features_path = args.features_path or os.path.join(features_dir, "features.npz")

    # Load features
    print("Loading features...")
    color, lbp, shape, y = load_features(features_path)
    X_std, _, _, _ = standardize_features(color, lbp, shape)

    # Get fold files
    train_files, val_files = get_fold_files(folds_dir)

    # Construct test file path (assuming test files follow the same pattern)
    test_file = os.path.join(folds_dir, f"test_fold_{args.fold_idx}.csv")

    # Load test data
    test_df = pd.read_csv(test_file)

    # Get indices for test data
    test_indices = [i for i, name in enumerate(test_df['filename']) if name in test_df['filename'].values]

    # Extract test data
    X_test = X_std[test_indices]
    y_test = y[test_indices]

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = lgb.Booster(model_file=args.model_path)

    # Make predictions
    print("Making predictions...")
    logits = model.predict(X_test)
    preds = np.argmax(logits, axis=1)

    # Evaluate
    accuracy = accuracy_score(y_test, preds)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Save predictions
    if args.output_file:
        output_path = args.output_file
    else:
        fold_output_dir = os.path.join(output_dir, f"fold_{args.fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)
        output_path = os.path.join(fold_output_dir, "test_predictions.csv")

    save_preds_logits(
        names=test_df['filename'].values,
        labels=y_test,
        preds=preds,
        logits=logits,
        output_file=output_path
    )
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
