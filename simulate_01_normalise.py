import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse

def normalize_and_save(data_dir, overwrite=0):
    os.makedirs(os.path.join(data_dir, 'normalized'), exist_ok=True)
    # Loop through all files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith("_blocks.csv"):
            file_path = os.path.join(data_dir, filename)
            
            # Construct the output file names
            global_norm_filename = filename.replace("_blocks.csv", "_global_norm.csv")
            strat_norm_filename = filename.replace("_blocks.csv", "_strat_norm.csv")
            
            global_norm_path = os.path.join(data_dir, 'normalized', global_norm_filename)
            strat_norm_path = os.path.join(data_dir, 'normalized', strat_norm_filename)

            # Check if files already exist
            if not overwrite and os.path.exists(global_norm_path) and os.path.exists(strat_norm_path):
                print(f"Files {global_norm_filename} and {strat_norm_filename} already exist. Skipping normalization.")
                continue  # Skip this file if both exist and overwrite is 0

            # Load the CSV file
            data = pd.read_csv(file_path, sep=",", index_col=0)
            
            # Perform global normalization
            data_GlobalNormalization = (data - data.mean()) / data.std()
            
            # Save global normalization result
            data_GlobalNormalization.to_csv(global_norm_path, sep=",")
            print(f"Global normalization saved to {global_norm_path}")

            # Relevant columns excluding 'Site'
            relevant_cols = list(set(data.columns) - set(["Site", "ID"]))

            # Initialize StandardScaler
            scaler = StandardScaler()

            # Apply stratified normalization with exception for all-NaN subgroups
            def stratified_normalization(group):
                if group[relevant_cols].isna().all().all():  # Check if all values are NaN in the relevant columns
                    return pd.DataFrame(np.nan, columns=relevant_cols, index=group.index)
                else:
                    # Perform normalization if the group is not full of NaNs
                    return pd.DataFrame(scaler.fit_transform(group[relevant_cols]), 
                                        columns=relevant_cols, 
                                        index=group.index)

            # Perform group normalization by 'Site', leaving NaN subgroups as is
            data_StratNormalization = data.groupby('Site', group_keys=False)[relevant_cols].apply(stratified_normalization)
            data_StratNormalization = pd.concat([data['Site'],data_StratNormalization],axis=1)

            # data_StratNormalization = data.groupby('Site', group_keys=False).apply(stratified_normalization)

            # # Add back 'ID' and 'Site' columns to the normalized DataFrame
            # data["ID"] = data.index
            # data_StratNormalization = data[['ID', 'Site']].join(data_StratNormalization)
            
            # Save stratified normalization result
            data_StratNormalization.to_csv(strat_norm_path, sep=",")
            print(f"Stratified normalization saved to {strat_norm_path}")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Normalize CSV files with global and stratified normalization.")
    
    # Add arguments
    parser.add_argument("data_dir", type=str, help="Directory where the *_blocks.csv files are located")
    parser.add_argument("--overwrite", type=int, default=0, help="Set to 1 to recompute normalizations even if files exist (default: 0)")

    # Parse arguments
    args = parser.parse_args()

    # Call the normalization function with parsed arguments
    print("I'm here no problem")
    normalize_and_save(args.data_dir, args.overwrite)