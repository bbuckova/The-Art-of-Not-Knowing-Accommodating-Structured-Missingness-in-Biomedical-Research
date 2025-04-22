import os
import pandas as pd
import argparse

def concatenate_files_with_seed(parent_dir, file_pattern):
    # Ensure output directory exists
    output_dir = os.path.join(parent_dir, "concatenated")
    os.makedirs(output_dir, exist_ok=True)

    # Process seed directories
    seed_dirs = [d for d in os.listdir(parent_dir) if d.startswith("seed_") and os.path.isdir(os.path.join(parent_dir, d))]
    if not seed_dirs:
        print("No seed directories found.")
        return

    # Find all unique base filenames in the first seed directory
    first_seed_dir = os.path.join(parent_dir, seed_dirs[0])
    base_files = [f for f in os.listdir(first_seed_dir) if f.endswith(file_pattern)]

    for base_file in base_files:

        if file_pattern == "_bootstrap":
            # Extract the base name without the seed-specific part
            base_name = base_file.split("_seed")[0]
            output_file = os.path.join(output_dir, f"{base_name}_concatenated.csv")
        elif file_pattern == "_imputed.csv":
            base_name = base_file.split("_imputed.csv")[0]
            output_file = os.path.join(output_dir, f"{base_name}_concatenated.csv")

        # Initialize an empty list to store DataFrames
        combined_data = []

        for seed_dir in seed_dirs:
            if file_pattern == "_imputed.csv":
                seed_number = seed_dir.split("seed_")[-1]
                seed_path = os.path.join(parent_dir, seed_dir)
                input_file = os.path.join(seed_path, base_file)

                if os.path.isfile(input_file):
                    try:
                        # Load the CSV and add the seed column
                        df = pd.read_csv(input_file)
                        df["seed"] = seed_number
                        combined_data.append(df)
                    except Exception as e:
                        print(f"Error processing {input_file}: {e}")
                else:
                    print(f"File {input_file} not found, skipping...")
            
            if file_pattern == "_bootstrap":
                seed_number = seed_dir.split("seed_")[-1]
                seed_path = os.path.join(parent_dir, seed_dir)
                input_file = os.path.join(seed_path, f"{base_file.split('seed')[0]}seed{seed_number}_bootstrap")

                if os.path.isfile(input_file):
                    try:
                        # Load the CSV and add the seed column
                        df = pd.read_csv(input_file)
                        df["seed"] = seed_number
                        combined_data.append(df)
                    except Exception as e:
                        print(f"Error processing {input_file}: {e}")
                else:
                    print(f"File {input_file} not found, skipping...")

        # Concatenate all DataFrames and save to the output file
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(f"Saved concatenated file: {output_file}")
        else:
            print(f"No valid files found for base name: {base_name}")



if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Concatenate files with seed information.")
    parser.add_argument("--parent_dir", required=True, help="Parent directory containing seed folders.")
    parser.add_argument("--method", required=True, choices=["AC", "ET"], help="Imputation method (MICE or ET).")
    args = parser.parse_args()

    # Map method to file pattern
    file_patterns = {
        "AC": "_bootstrap",
        "ET": "_imputed.csv"
    }
    file_pattern = file_patterns[args.method]

    # Call the main function
    concatenate_files_with_seed(args.parent_dir, file_pattern)
