import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from configs.config import load_yaml_file
from src.models.lightgbm_model import load_features, get_fold_files, standardize_features, fit_and_predict_lgbm, save_preds_logits
from src.utils.lgbm_feature_extract import extract_and_save_features, generate_masks

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model for mold classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--extract_features", action="store_true", help="Extract features before training")
    parser.add_argument("--fold_idx", type=int, default=None, help="If set, trains only this fold; otherwise all folds")
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_file(args.config)
    
    # Set up directories
    data_dir = config.get("DATA_DIR", "data")
    features_dir = config.get("FEATURES_DIR", "features")
    output_dir = config.get("OUTPUT_DIR", "output")
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    features_path = os.path.join(features_dir, "features.npz")
    folds_dir = os.path.join(data_dir, "folds")
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features if requested
    if args.extract_features:
        print("Generating masks...")
        generate_masks(image_dir, mask_dir)
        
        print("Extracting features...")
        labels_csv = os.path.join(data_dir, "dataset.csv")
        extract_and_save_features(
            image_dir=image_dir,
            mask_dir=mask_dir,
            labels_csv=labels_csv,
            output_path=features_path,
            image_size=(256, 256)
        )
    
    # Load features
    print("Loading features...")
    color, lbp, shape, y = load_features(features_path)
    X_std, scaler_color, scaler_lbp, scaler_shape = standardize_features(color, lbp, shape)
    
    # Get fold files
    train_files, val_files = get_fold_files(folds_dir)
    
    # Determine which folds to train
    if args.fold_idx is not None:
        fold_indices = [args.fold_idx]
    else:
        fold_indices = range(len(train_files))
    
    # LightGBM parameters from config
    lgbm_params = {
        'num_class': len(config.get("CLASS_NAMES", 5)),
        'num_leaves': config.get("LGB_MAX_DEPTH", 7),
        'learning_rate': config.get("LGB_LEARNING_RATE", 0.05),
        'feature_fraction': 0.9
    }
    
    # Train and evaluate for each fold
    for fold_idx in fold_indices:
        print(f"\n===== Fold {fold_idx} =====")
        
        # Load fold data
        train_df = pd.read_csv(train_files[fold_idx])
        val_df = pd.read_csv(val_files[fold_idx])
        
        # Get indices for this fold
        train_indices = [i for i, name in enumerate(train_df['filename']) if name in train_df['filename'].values]
        val_indices = [i for i, name in enumerate(val_df['filename']) if name in val_df['filename'].values]
        
        # Extract data for this fold
        X_train = X_std[train_indices]
        y_train = y[train_indices]
        X_val = X_std[val_indices]
        y_val = y[val_indices]
        
        # Train model
        print("Training LightGBM model...")
        preds, logits, model = fit_and_predict_lgbm(
            X_train, y_train, X_val, 
            num_class=lgbm_params['num_class'],
            num_leaves=lgbm_params['num_leaves'],
            learning_rate=lgbm_params['learning_rate'],
            feature_fraction=lgbm_params['feature_fraction']
        )
        
        # Evaluate
        accuracy = accuracy_score(y_val, preds)
        print(f"Validation accuracy: {accuracy:.4f}")
        print(classification_report(y_val, preds))
        
        # Save predictions
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        save_preds_logits(
            names=val_df['filename'].values,
            labels=y_val,
            preds=preds,
            logits=logits,
            output_file=os.path.join(fold_output_dir, "val_predictions.csv")
        )
        
        # Save model
        model.save_model(os.path.join(fold_output_dir, "model.txt"))
        print(f"Model saved to {fold_output_dir}/model.txt")

if __name__ == "__main__":
    main()