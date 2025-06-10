import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from configs.config import load_yaml_file
from src.models import MoldVision, MoldVGG_6Chan, MoldVGG_Default
from src.databases import MoldDataModule

def main():
    parser = argparse.ArgumentParser(description="Train VGG model for mold classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model", type=str, choices=["twin", "6chan", "default"], required=True, 
                        help="Model type: twin (MoldVision), 6chan, or default")
    parser.add_argument("--fold_idx", type=int, default=None, 
                        help="If set, trains only this fold; otherwise all folds")
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_file(args.config)
    
    # Set up directories
    data_dir = config.get("DATA_DIR", "data")
    output_dir = config.get("OUTPUT_DIR", "output")
    log_dir = config.get("LOG_DIR", "logs")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get hyperparameters from config
    batch_size = config.get("BATCH_SIZE", 32)
    num_epochs = config.get("NUM_EPOCHS", 50)
    learning_rate = config.get("LEARNING_RATE", 0.001)
    num_folds = config.get("N_FOLDS", 5)
    num_classes = len(config.get("CLASS_NAMES", ["peni", "aspfum", "aspfla", "clado", "fus"]))
    device = config.get("DEVICE", "cuda")
    
    # Determine which folds to train
    if args.fold_idx is not None:
        fold_indices = [args.fold_idx]
    else:
        fold_indices = range(num_folds)
    
    # Train each fold
    for fold_idx in fold_indices:
        print(f"\n===== Fold {fold_idx}/{num_folds-1} =====")
        
        # Set up data module
        dm = MoldDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_folds=num_folds,
            fold_idx=fold_idx,
            num_workers=0,
            pin_memory=True,
            transforms=None,  # Add transforms if needed
            model_type=args.model
        )
        dm.prepare_data()
        dm.setup()
        
        # Initialize model
        if args.model == "twin":
            model = MoldVision(opt_lr=learning_rate, lr_pat=5, out_features=num_classes)
        elif args.model == "6chan":
            model = MoldVGG_6Chan(opt_lr=learning_rate, lr_pat=5, out_features=num_classes)
        else:  # default
            model = MoldVGG_Default(opt_lr=learning_rate, lr_pat=5, out_features=num_classes)
        
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, f"fold_{fold_idx}"),
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        )
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
        
        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu" if device == "cuda" and pl.utilities.rank_zero_only.rank_zero_only() else "cpu",
            devices=1,
            default_root_dir=os.path.join(log_dir, f"fold_{fold_idx}"),
            callbacks=[checkpoint_callback, early_stop_callback]
        )
        
        # Train model
        trainer.fit(model, datamodule=dm)
        
        # Test model
        trainer.test(model, datamodule=dm)
        
        # Save final model
        trainer.save_checkpoint(os.path.join(output_dir, f"fold_{fold_idx}", "final_model.ckpt"))
        print(f"Model saved to {os.path.join(output_dir, f'fold_{fold_idx}', 'final_model.ckpt')}")

if __name__ == "__main__":
    main()