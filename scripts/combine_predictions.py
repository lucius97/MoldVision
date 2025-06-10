import os
import argparse
import pandas as pd
import glob

def combine_prediction_files(input_dir, output_file, model_type=None, file_pattern=None):
    """
    Combine prediction CSV files from multiple folds into a single CSV file.
    
    Args:
        input_dir (str): Directory containing fold subdirectories with prediction files
        output_file (str): Path to save the combined CSV file
        model_type (str, optional): Type of model ('lgbm' or 'vgg'). If provided, will look for specific patterns.
        file_pattern (str, optional): Custom file pattern to match. If not provided, uses default patterns.
    """
    # Determine file pattern to search for
    if file_pattern is None:
        if model_type == 'lgbm':
            file_pattern = "fold_*/test_predictions.csv"
        elif model_type == 'vgg':
            file_pattern = "fold_*/test_predictions.csv"
        else:
            file_pattern = "fold_*/test_predictions.csv"  # Default pattern
    
    # Find all matching files
    search_pattern = os.path.join(input_dir, file_pattern)
    prediction_files = glob.glob(search_pattern)
    
    if not prediction_files:
        print(f"No prediction files found matching pattern: {search_pattern}")
        return False
    
    print(f"Found {len(prediction_files)} prediction files")
    
    # Read and combine all files
    dfs = []
    for file in prediction_files:
        fold = os.path.basename(os.path.dirname(file)).replace("fold_", "")
        df = pd.read_csv(file)
        df['fold'] = fold  # Add fold information
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataframe
    combined_df.to_csv(output_file, index=False)
    print(f"Combined predictions saved to {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Combine prediction CSVs from multiple folds")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing fold subdirectories")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save combined CSV")
    parser.add_argument("--model_type", type=str, choices=['lgbm', 'vgg'], help="Type of model")
    parser.add_argument("--file_pattern", type=str, help="Custom file pattern to match")
    args = parser.parse_args()
    
    combine_prediction_files(args.input_dir, args.output_file, args.model_type, args.file_pattern)

if __name__ == "__main__":
    main()