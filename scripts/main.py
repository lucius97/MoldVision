import argparse
import os
import subprocess
from configs.config import load_yaml_file

def run_lgbm_train(config_path, fold_idx=None, extract_features=False):
    """Run LightGBM training script"""
    cmd = ["python", "scripts/train_lgbm.py", "--config", config_path]

    if extract_features:
        cmd.append("--extract_features")

    if fold_idx is not None:
        cmd.extend(["--fold_idx", str(fold_idx)])

    print("[INFO] Running LightGBM training")
    subprocess.run(cmd)

def run_lgbm_test(config_path, model_path, fold_idx=0, output_file=None):
    """Run LightGBM testing script"""
    cmd = ["python", "scripts/test_lgbm.py", 
           "--config", config_path,
           "--model_path", model_path,
           "--fold_idx", str(fold_idx)]

    if output_file:
        cmd.extend(["--output_file", output_file])

    print("[INFO] Running LightGBM testing")
    subprocess.run(cmd)

def run_vgg_train(config_path, model_type, fold_idx=None):
    """Run VGG training script"""
    # Map model names from CLI to actual model types
    model_map = {
        "default": "default",
        "sixchan": "6chan",
        "vision": "twin"
    }

    model_arg = model_map.get(model_type, "default")

    cmd = ["python", "scripts/train_vgg.py", 
           "--config", config_path,
           "--model", model_arg]

    if fold_idx is not None:
        cmd.extend(["--fold_idx", str(fold_idx)])

    print(f"[INFO] Running VGG training: {model_type}")
    subprocess.run(cmd)

def run_vgg_test(config_path, model_path, model_type, fold_idx=0, output_file=None):
    """Run VGG testing script"""
    # Map model names from CLI to actual model types
    model_map = {
        "default": "default",
        "sixchan": "6chan",
        "vision": "twin"
    }

    model_arg = model_map.get(model_type, "default")

    cmd = ["python", "scripts/test_vgg.py", 
           "--config", config_path,
           "--model_path", model_path,
           "--model_type", model_arg,
           "--fold_idx", str(fold_idx)]

    if output_file:
        cmd.extend(["--output_file", output_file])

    print(f"[INFO] Running VGG testing: {model_type}")
    subprocess.run(cmd)

def run_combine_predictions(input_dir, output_file, model_type=None, file_pattern=None):
    """Run the combine_predictions script to combine CSVs from multiple folds"""
    cmd = ["python", "scripts/combine_predictions.py", 
           "--input_dir", input_dir,
           "--output_file", output_file]

    if model_type:
        cmd.extend(["--model_type", model_type])

    if file_pattern:
        cmd.extend(["--file_pattern", file_pattern])

    print(f"[INFO] Combining prediction files from {input_dir}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Unified launcher for mold classification models.")

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # LightGBM workflow
    lgbm_parser = subparsers.add_parser("lgbm", help="Run LightGBM workflow")
    lgbm_subparsers = lgbm_parser.add_subparsers(dest="lgbm_mode", required=True)

    # LightGBM train
    lgbm_train_parser = lgbm_subparsers.add_parser("train", help="Train LightGBM model")
    lgbm_train_parser.add_argument("--extract_features", action="store_true", help="Extract features before training")
    lgbm_train_parser.add_argument("--fold_idx", type=int, help="If set, trains only this fold")

    # LightGBM test
    lgbm_test_parser = lgbm_subparsers.add_parser("test", help="Test LightGBM model")
    lgbm_test_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file")
    lgbm_test_parser.add_argument("--fold_idx", type=int, default=0, help="Fold index to test")
    lgbm_test_parser.add_argument("--output_file", type=str, help="Path to save predictions")

    # Neural network workflow
    nn_parser = subparsers.add_parser("neural", help="Run neural network workflow")
    nn_parser.add_argument(
        "--model",
        type=str,
        choices=["default", "sixchan", "vision"],
        required=True,
        help="Which neural model to use: default, sixchan, or vision"
    )

    nn_subparsers = nn_parser.add_subparsers(dest="nn_mode", required=True)

    # Neural train
    nn_train_parser = nn_subparsers.add_parser("train", help="Train neural model")
    nn_train_parser.add_argument("--fold_idx", type=int, help="If set, trains only this fold")

    # Neural test
    nn_test_parser = nn_subparsers.add_parser("test", help="Test neural model")
    nn_test_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    nn_test_parser.add_argument("--fold_idx", type=int, default=0, help="Fold index to test")
    nn_test_parser.add_argument("--output_file", type=str, help="Path to save predictions")

    # Combine predictions command
    combine_parser = subparsers.add_parser("combine", help="Combine prediction CSVs from multiple folds")
    combine_parser.add_argument("--input_dir", type=str, required=True, help="Directory containing fold subdirectories")
    combine_parser.add_argument("--output_file", type=str, required=True, help="Path to save combined CSV")
    combine_parser.add_argument("--model_type", type=str, choices=['lgbm', 'vgg'], help="Type of model")
    combine_parser.add_argument("--file_pattern", type=str, help="Custom file pattern to match")

    args = parser.parse_args()

    # Load config
    config_path = args.config

    if args.command == "lgbm":
        if args.lgbm_mode == "train":
            run_lgbm_train(config_path, args.fold_idx, args.extract_features)
        elif args.lgbm_mode == "test":
            run_lgbm_test(config_path, args.model_path, args.fold_idx, args.output_file)
        else:
            lgbm_parser.print_help()

    elif args.command == "neural":
        if args.nn_mode == "train":
            run_vgg_train(config_path, args.model, args.fold_idx)
        elif args.nn_mode == "test":
            run_vgg_test(config_path, args.model_path, args.model, args.fold_idx, args.output_file)
        else:
            nn_parser.print_help()

    elif args.command == "combine":
        run_combine_predictions(args.input_dir, args.output_file, args.model_type, args.file_pattern)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
