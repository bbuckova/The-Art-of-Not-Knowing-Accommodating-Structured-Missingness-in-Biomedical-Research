import os
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def ExtraTrees_Imputation(target_dir, complete_file, missing_file, seed=42):
    """
    Perform imputation on a dataset with missing values using the ExtraTreesRegressor.
    
    Parameters:
    - target_dir (str): Directory to save the imputed output file.
    - complete_file (str): Path to the file containing the complete dataset.
    - missing_file (str): Path to the file containing the dataset with missing values.
    - seed (int, optional): Random seed for reproducibility. Default is 42.
    
    Returns:
    - Saves an imputed CSV file in the specified target directory.
    """
    
    # Load datasets
    blocks = pd.read_csv(missing_file, sep=',', index_col=0)
    complete = pd.read_csv(complete_file, sep=',', index_col=0)
    
    # Define ExtraTrees as the estimator for IterativeImputer
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(
                                                                n_estimators=10, 
                                                                random_state=seed, 
                                                                max_depth = 5, 
                                                                max_samples = int(round(0.1*complete.shape[0])), 
                                                                bootstrap=True), 
                               max_iter=100, random_state=seed)
    
    # Fit and transform the data
    imputed_data = imputer.fit_transform(blocks)
    
    # Convert the result back to a DataFrame
    t_name = os.path.basename(complete_file).replace('_complete', '_imputed')

    imputed_df = pd.DataFrame(imputed_data, columns=complete.columns)
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Save the imputed data
    output_path = os.path.join(target_dir, t_name)
    imputed_df.to_csv(output_path, index=True)
    print(f"Imputed data saved to: {output_path}")


# To run from the command line, uncomment the following lines:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ExtraTrees-based imputation on a dataset.")
    parser.add_argument("target_dir", type=str, help="Directory to save the imputed output file")
    parser.add_argument("complete_file", type=str, help="Path to the complete dataset file")
    parser.add_argument("missing_file", type=str, help="Path to the dataset file with missing values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    ExtraTrees_Imputation(args.target_dir, args.complete_file, args.missing_file, args.seed)
