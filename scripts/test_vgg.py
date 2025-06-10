import os
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from configs.config import load_yaml_file
from src.models import MoldVision, MoldVGG_6Chan, MoldVGG_Default
from src.databases import MoldDataModule

def main():
    parser = argparse.ArgumentParser(description="Test VGG model for mold classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["twin", "6chan", "default"], required=True, 
                        help="Model type: twin (MoldVision), 6chan, or default")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index to test")
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_file(args.config)

    # Set up directories
    data_dir = config.get("DATA_DIR", "data")
    output_dir = config.get("OUTPUT_DIR", "output")

    # Get hyperparameters from config
    batch_size = config.get("INFER_BATCH_SIZE", 64)
    num_folds = config.get("N_FOLDS", 5)
    num_classes = len(config.get("CLASS_NAMES", ["peni", "aspfum", "aspfla", "clado", "fus"]))
    device = config.get("DEVICE", "cuda")

    # Set up data module for testing
    dm = MoldDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_folds=num_folds,
        fold_idx=args.fold_idx,
        num_workers=0,
        pin_memory=True,
        transforms=None,  # Add transforms if needed
        model_type=args.model_type
    )
    dm.prepare_data()
    dm.setup(stage='test')

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.model_type == "twin":
        model = MoldVision.load_from_checkpoint(args.model_path)
    elif args.model_type == "6chan":
        model = MoldVGG_6Chan.load_from_checkpoint(args.model_path)
    else:  # default
        model = MoldVGG_Default.load_from_checkpoint(args.model_path)

    # Set up trainer for testing
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
    )

    # Test model
    print("Testing model...")
    test_results = trainer.test(model, datamodule=dm)

    # Get predictions
    print("Getting predictions...")
    predictions = []
    true_labels = []
    filenames = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        for batch in dm.test_dataloader():
            if args.model_type == "twin":
                x, y, fname = batch
                logits = model(x[0], x[1])
            elif args.model_type == "6chan":
                x, y, fname = batch
                logits = model(x)
            else:  # default
                x, y, fname = batch
                logits = model(x, x)  # Default model expects two inputs

            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            filenames.extend(fname)
            all_logits.append(logits.cpu().numpy())

    # Concatenate logits from all batches
    all_logits = np.vstack(all_logits)

    # Evaluate
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

    # Save predictions
    if args.output_file:
        output_path = args.output_file
    else:
        fold_output_dir = os.path.join(output_dir, f"fold_{args.fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)
        output_path = os.path.join(fold_output_dir, "test_predictions.csv")

    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        'name': filenames,
        'label': true_labels,
        'pred': predictions
    })

    # Add logits for each class
    for i in range(all_logits.shape[1]):
        results_df[f'logit_{i}'] = all_logits[:, i]

    # Map numeric labels to class names if available
    if "CLASS_NAMES" in config:
        class_names = config["CLASS_NAMES"]
        results_df['true_class'] = results_df['label'].apply(lambda x: class_names[x] if x < len(class_names) else str(x))
        results_df['predicted_class'] = results_df['pred'].apply(lambda x: class_names[x] if x < len(class_names) else str(x))

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
