import os
import argparse
import numpy as np
import pandas as pd
import pyarrow
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def restore_global_normalization(data_GlobalNormalization, original_data):
    """
    Restores the original values from globally normalized data.

    Parameters:
    - data_GlobalNormalization (pd.DataFrame): The globally normalized dataframe.
    - original_data (pd.DataFrame): The original data before normalization.

    Returns:
    - pd.DataFrame: Restored data with original values.
    """
    restored_data = data_GlobalNormalization * original_data.std() + original_data.mean()
    return restored_data

def restore_stratified_normalization(data_StratNormalization, original_data):
    """
    Restores the original values from stratified normalized data.

    Parameters:
    - data_StratNormalization (pd.DataFrame): The stratified normalized dataframe.
    - original_data (pd.DataFrame): The original data before normalization.

    Returns:
    - pd.DataFrame: Restored data with original values.
    """
    relevant_cols = list(set(original_data.columns) - set(["Site", "ID"]))
    restored_data = pd.DataFrame()

    for site, group in original_data.groupby('Site'):
        if not group[relevant_cols].isna().all().all():
            restored_group = data_StratNormalization[data_StratNormalization['Site'] == site].copy()
            restored_group[relevant_cols] = restored_group[relevant_cols] * group[relevant_cols].std() + group[relevant_cols].mean()
            restored_data = pd.concat([restored_data, restored_group])
        else:
            restored_group = data_StratNormalization[data_StratNormalization['Site'] == site].copy()
            restored_data = pd.concat([restored_data, restored_group])
    
    return restored_data

def normalize_to_probability(p, q, nbins=0.5):
    """
    Transforms two input vectors into probability distributions using a common histogram binning.
    
    Parameters:
    - p (numpy array): First input vector.
    - q (numpy array): Second input vector.
    - nbins (percentage 0-1): smoothness of the histogram (as percentage of the original vector length)
    
    Returns:
    - tuple of numpy arrays: (prob_dist_p, prob_dist_q) 
        - prob_dist_p (numpy array): Probability distribution for vector p.
        - prob_dist_q (numpy array): Probability distribution for vector q.
    
    Notes:
    - The function calculates a common range based on the minimum and maximum values 
      across both vectors, then creates a fixed number of bins to construct histograms 
      for each vector. 
    - Small positive values (epsilon) are added to avoid division by zero.
    """
    # Define the number of bins you want
    num_bins = round(len(p) * nbins)

    # Find the min and max values across both arrays to create common bins
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())

    # Create histogram bins
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Compute the histograms and normalize to form probability distributions
    hist_p, _ = np.histogram(p, bins=bins)
    hist_q, _ = np.histogram(q, bins=bins)
    
    # Avoid zero values for stable probability distribution calculation
    epsilon = 1e-10
    hist_p_clipped = np.clip(hist_p, epsilon, None)
    hist_q_clipped = np.clip(hist_q, epsilon, None)
    prob_dist_p = hist_p_clipped / hist_p_clipped.sum()
    prob_dist_q = hist_q_clipped / hist_q_clipped.sum()

    return prob_dist_p, prob_dist_q


def evaluate_distributions(sn_scaled, gt, blocks, cols_to_eval, param_dict):
    """
    Evaluates KL divergence, JS divergence, and NMSE for all columns in cols_to_eval between two datasets.

    Parameters:
    - sn_scaled (pd.DataFrame): Imputed data (after restoration).
    - gt (pd.DataFrame): Ground truth data.
    - blocks (pd.DataFrame): Mask/block data to mask missing values.
    - cols_to_eval (list): List of columns to evaluate.
    - param_dict (dict): Dictionary containing parsed parameters from the filename.

    Returns:
    - pd.DataFrame: Evaluation results including KL divergence, JS divergence, NMSE, and distance from (1,1).
    """
    results = {}

    for col in cols_to_eval:
        p = sn_scaled.mask(blocks.notna())[col].dropna().to_numpy()
        q = gt.mask(blocks.notna())[col].dropna().to_numpy()

        if len(p) == 0 or len(q) == 0:
            continue

        p_norm,q_norm = normalize_to_probability(p,q)

        kl = entropy(q_norm, qk=p_norm, base=2)
        js = 1 - jensenshannon(p_norm, q_norm, base=2)

        nom = ((q - p) ** 2).sum()
        denom = ((q - q.mean()) ** 2).sum()
        nmse = 1 - (nom / denom) if denom != 0 else np.nan
        wasserstein = wasserstein_distance(p, q)

        results[col] = {
            'KL Divergence': kl,
            '1-JS Divergence': js,
            'NMSE': nmse,
            'Wasserstein': wasserstein
        }

    results_df = pd.DataFrame(results).T
    point = np.array([1, 1])
    results_df['NMSE_JS_combined'] = np.sqrt((results_df["1-JS Divergence"] - point[0])**2 + (results_df['NMSE'] - point[1])**2)

    # Add the parameters as columns to the DataFrame
    for param, value in param_dict.items():
        results_df[param] = value

    return results_df

def parse_filename_for_params(filename):
    """
    Parses the filename to extract imputation parameters.

    Parameters:
    - filename (str): Filename from which to extract parameters.

    Returns:
    - dict: A dictionary containing the parsed parameters.
    """
    basename = os.path.basename(filename)
    # Example: "samples50_overlap1_noise0.1.csv_n50_mtreal_ol1_ds2_vl8_no0.1_se42.0_1_complete.csv"
    try:
        param_str = basename.split('_complete')[0]
        params = param_str.split('__')[1].split('_')

        param_dict = {}
        for param in params:
            if param.startswith('samples'):
                param_dict['num_samples'] = int(param.split('samples')[1])
            elif param.startswith('overlap'):
                param_dict['overlap'] = int(param.split('overlap')[1])
            elif param.startswith('noise0'):
                param_dict['noise'] = str(param.split('noise')[1])
            else:
                param_dict['missingness'] = param
                
        return param_dict

    except IndexError:
        raise ValueError("Filename format not recognized for extracting parameters.")

def calculate_coverage_rate(gn, gt, blocks, cols_to_eval, param_dict, confidence_level=0.95):
    """
    Computes the coverage rate (CR) and the width of confidence intervals (CI) for the given data.
    
    Parameters:
    ----------
    gn : DataFrame
        Imputed data containing IDs, seed values, and columns with variables to evaluate.
    gt : DataFrame
        Ground truth data with original values for comparison against imputed data.
    blocks : DataFrame
        Masking data indicating observed and missing values across columns and rows.
    cols_to_eval : list
        List of variable names (columns) to evaluate for CI coverage.
    confidence_level : float, optional
        Confidence level for interval calculation, default is 0.95.
    
    Returns:
    -------
    falls_to_CI : Series
        Coverage rate per variable, representing the proportion of values in the ground truth
        that fall within the computed confidence intervals.
    CI_width : float
        Mean width of confidence intervals, normalized by the standard deviation of each variable.
    """

    # Pivot and transform to 3D array
    pivoted_data = gn.pivot_table(index="ID", columns="seed", values=cols_to_eval)
    pivoted_data = pivoted_data.sort_index(axis=1, level=['seed'])[cols_to_eval]
    gn_array = pivoted_data.to_numpy().reshape(len(pivoted_data), len(cols_to_eval), -1)

    # Calculate percentiles for the confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_pct_values = np.percentile(gn_array, lower_percentile, axis=2)
    upper_pct_values = np.percentile(gn_array, upper_percentile, axis=2)
    
    # Convert results to DataFrames with appropriate index and columns
    lower_pct_values = pd.DataFrame(lower_pct_values, index=pivoted_data.index, columns=cols_to_eval)
    upper_pct_values = pd.DataFrame(upper_pct_values, index=pivoted_data.index, columns=cols_to_eval)
    
    # Calculate whether the original data falls within the confidence intervals
    falls_to_CI = (gt[cols_to_eval] > lower_pct_values) & (gt[cols_to_eval] < upper_pct_values)
    divide_by = (blocks[cols_to_eval].isna().sum())
    divide_by[divide_by ==0] =1
    falls_to_CI = (falls_to_CI.mask(blocks[cols_to_eval].notna()).sum()) / divide_by
    
    # Calculate CI width, normalized by standard deviation
    CI_width = ((upper_pct_values - lower_pct_values).mask(blocks.notna()) / gt[blocks.notna()].std()[cols_to_eval]).mean(axis=0, skipna=True)
    
    df_cv = pd.DataFrame([falls_to_CI, CI_width])
    df_cv = df_cv.T
    df_cv.columns = ["coverage","width"]
    df_cv = df_cv.assign(**param_dict)

    df_cv.loc[blocks.isna().sum(axis=0) == 0, 'coverage'] = np.nan
    df_cv.loc[blocks.isna().sum(axis=0) == 0, 'width'] = np.nan

    return df_cv.reset_index()

def load_and_clean(impute_path, variables, algorithm):    
    if algorithm == "AutoComplete":
        impute = pd.read_csv(impute_path, low_memory=False)
        impute = impute[impute["ID"] != "ID"]
        impute["ID"] = impute["ID"].astype(int)
        # impute["ID"] = impute["ID"].astype(str).str.isnumeric()
        impute = impute[impute["seed"].astype(str).str.isnumeric() & impute["ID"].astype(str).str.isnumeric()]
        impute["seed"] = impute["seed"].astype(int)

    elif (algorithm == "ET") | (algorithm == "Average"):
        impute = pd.read_csv(impute_path)
        impute = impute.rename(columns={"Unnamed: 0" : "ID"})
        impute = impute[impute["ID"].notna()]
        impute["ID"] = impute["ID"].astype(int)
        impute = impute[impute["seed"].astype(str).str.isnumeric() & impute["ID"].astype(str).str.isnumeric()]
        impute["seed"] = impute["seed"].astype(int)

    elif algorithm == "MICE":
        impute = pd.read_parquet(impute_path)
        impute = impute.rename(columns={".imp":"seed", ".id":"ID"})
        impute["ID"] = impute["ID"]-1

    impute[variables + ["Age", "Site"]] = impute[variables + ["Age", "Site"]].astype(float)
    return impute

def compare_matrices(complete, imputed, variables, algorithm, param_dict, imputed2=None):
    """
    Compare the correlation matrices of complete and imputed datasets for each unique seed.

    Parameters:
    - complete (pd.DataFrame): The dataset containing the complete (ground truth) data.
    - imputed (pd.DataFrame): The dataset containing the imputed data with a 'seed' column to distinguish imputations.
    - variables (list): List of column names to include in the correlation matrices, excluding "Age".

    Returns:
    - frobenius_distance (list): Frobenius distances between the upper triangles of the correlation matrices for each seed.
    - similarity (list): Pearson correlation coefficients between the upper triangles of the correlation matrices for each seed.

    Notes:
    - The function assumes that both the 'complete' and 'imputed' datasets have the specified variables and "Age" column.
    - The comparison uses the upper triangular parts of the correlation matrices, excluding the diagonal.
    """
    frobenius_distance = []
    similarity = []

    for iseed in imputed['seed'].unique():
        # Compute correlation matrices
        corr_original = complete[[*variables, "Age"]].corr()
        corr_imputed = imputed[imputed["seed"] == iseed][[*variables, "Age"]].corr()

        # Extract the upper triangular part, excluding the diagonal
        upper_original = corr_original.to_numpy()[np.triu_indices_from(corr_original, k=1)]
        upper_imputed = corr_imputed.to_numpy()[np.triu_indices_from(corr_imputed, k=1)]

        # Compute Frobenius norm (distance)
        frobenius_distance.append(np.linalg.norm(upper_original - upper_imputed))

        # Compute similarity (Pearson correlation of flattened triangles)
        similarity.append(np.corrcoef(upper_original, upper_imputed)[0, 1])

    df_fs = pd.DataFrame([frobenius_distance, similarity]).T
    df_fs.columns = ["Frobenius", "Correlation"]
    df_fs["seed"] = np.arange(1,11)

    if algorithm == "AutoComplete":
        df_fs["normalisation"] = "global"
    else:
        df_fs["normalisation"] = 'none'

    if isinstance(imputed2, pd.DataFrame):
        frobenius_distance = []
        similarity = []
    
        for iseed in imputed['seed'].unique():
            # Compute correlation matrices
            corr_original = complete[[*variables, "Age"]].corr()
            corr_imputed = imputed[imputed["seed"] == iseed][[*variables, "Age"]].corr()

            # Extract the upper triangular part, excluding the diagonal
            upper_original = corr_original.to_numpy()[np.triu_indices_from(corr_original, k=1)]
            upper_imputed = corr_imputed.to_numpy()[np.triu_indices_from(corr_imputed, k=1)]

            # Compute Frobenius norm (distance)
            frobenius_distance.append(np.linalg.norm(upper_original - upper_imputed))

            # Compute similarity (Pearson correlation of flattened triangles)
            similarity.append(np.corrcoef(upper_original, upper_imputed)[0, 1])

        df_fs2 = pd.DataFrame([frobenius_distance, similarity]).T
        df_fs2.columns = ["Frobenius", "Correlation"]
        df_fs2["normalisation"] = "stratified"

        df_fs = pd.concat([df_fs, df_fs2])
    
    df_fs = df_fs.assign(**param_dict)

    return df_fs


def main(algorithm, complete_data, blocks_data, global_norm_imputed=None, strat_norm_imputed=None, imputed=None, output_dir=None):
    file_name = os.path.basename(complete_data).removesuffix("_complete.csv")
    gt = pd.read_table(complete_data, sep=',', index_col=0)
    blocks = pd.read_table(blocks_data, sep=',', index_col=0)
    variables = [f"Variable_{i}" for i in range(1, 41)]
    
    if algorithm == "AutoComplete":
        gn_impute = load_and_clean(global_norm_imputed, variables, algorithm)
        sn_impute = load_and_clean(strat_norm_imputed, variables, algorithm)

        cols_to_eval = sorted(
            [col for col in blocks.columns if col not in ["Age", "Site"] and blocks[col].notna().any()],
            key=lambda x: int(x.split('_')[1])
        )

        gn_normalized_frames, sn_normalized_frames, all_results = [], [], []
        for iseed in sorted(gn_impute["seed"].unique()):
            print(f"Processing seed: {iseed}")
            gn_impute_iseed = gn_impute[gn_impute["seed"] == iseed].set_index("ID")
            sn_impute_iseed = sn_impute[sn_impute["seed"] == iseed].set_index("ID")

            gn_scaled = restore_global_normalization(gn_impute_iseed, gt)
            sn_scaled = restore_stratified_normalization(sn_impute_iseed, gt)
            gn_scaled, sn_scaled = gn_scaled[cols_to_eval], sn_scaled[cols_to_eval]
            gn_scaled["seed"], sn_scaled["seed"] = iseed, iseed
            gn_scaled["ID"], sn_scaled["ID"] = gn_impute_iseed.index, sn_impute_iseed.index

            gn_normalized_frames.append(gn_scaled)
            sn_normalized_frames.append(sn_scaled)
            
            param_dict = parse_filename_for_params(complete_data)
            results_sn = evaluate_distributions(sn_scaled, gt, blocks, cols_to_eval, param_dict)
            results_sn["normalisation"], results_sn["seed"] = "stratified", iseed
            results_gn = evaluate_distributions(gn_scaled, gt, blocks, cols_to_eval, param_dict)
            results_gn["normalisation"], results_gn["seed"] = "global", iseed
            all_results.extend([results_sn, results_gn])

        gn = pd.concat(gn_normalized_frames, ignore_index=True)
        sn = pd.concat(sn_normalized_frames, ignore_index=True)
        final_results = pd.concat(all_results, ignore_index=False).reset_index()
        df_gn_cv = calculate_coverage_rate(gn, gt, blocks, cols_to_eval, param_dict, confidence_level=0.95)
        df_gn_cv["normalisation"] = "global"
        df_sn_cv = calculate_coverage_rate(sn, gt, blocks, cols_to_eval, param_dict, confidence_level=0.95)
        df_sn_cv["normalisation"] = "stratified"
        df_cv = pd.concat([df_sn_cv.reset_index(),df_gn_cv.reset_index()], axis=0)

        df_fs = compare_matrices(gt, gn_impute, variables, algorithm, param_dict, imputed2=sn_impute)

        output_dir = os.path.join(output_dir,"quality")
        os.makedirs(output_dir, exist_ok=True)
        final_results.to_csv(os.path.join(output_dir, file_name + "_quality.csv"), index=False)
        df_cv.to_csv(os.path.join(output_dir, file_name + "_coverage.csv"), index=False)
        df_fs.to_csv(os.path.join(output_dir, file_name + "_similarities.csv"), index=False)

    elif algorithm == "ET":
        # imputed_df = pd.read_table(imputed, sep=',', index_col=0)
        df_impute = load_and_clean(imputed, variables, algorithm)
        cols_to_eval = sorted(        
        [col for col in blocks.columns if col not in ["Age", "Site"] and blocks[col].notna().any()],
        key=lambda x: int(x.split('_')[1]))

        df_results = []
        for iseed in sorted(df_impute["seed"].unique()):
            df_impute_iseed = df_impute[df_impute["seed"] == iseed].set_index("ID")
            param_dict = parse_filename_for_params(complete_data)
            results = evaluate_distributions(df_impute_iseed, gt, blocks, cols_to_eval, param_dict)
            results["normalisation"] = "none"
            results["seed"] = iseed
            df_results.append(results)
        
        df_results = pd.concat(df_results, ignore_index=False)
        df_cv = calculate_coverage_rate(df_impute, gt, blocks, cols_to_eval, param_dict, confidence_level=0.95)
        df_fs = compare_matrices(gt, df_impute, variables, algorithm, param_dict)

        quality_dir = os.path.join(os.path.dirname(imputed), 'quality')
        os.makedirs(quality_dir, exist_ok=True)
        
        df_results.to_csv(os.path.join(quality_dir, file_name + "_quality.csv"))
        # df_results.to_csv(os.path.join(output_dir, file_name + "_combined_results.csv"), index=False)
        df_cv.to_csv(os.path.join(quality_dir, file_name + "_coverage.csv"))
        df_fs.to_csv(os.path.join(quality_dir, file_name + "_similarities.csv"))
        
        print(f"Imputation quality metrics saved to {os.path.join(quality_dir, file_name + '_quality.csv')}")
        print(f"Imputation quality metrics saved to {os.path.join(quality_dir, file_name + '_similarities.csv')}")
        
    elif algorithm == "MICE":
        df_impute = load_and_clean(imputed, variables, algorithm)

        cols_to_eval = sorted(        
                                [col for col in blocks.columns if col not in ["Age", "Site"] and blocks[col].notna().any()],
                                key=lambda x: int(x.split('_')[1]))
            
        df_results = []
        for iseed in sorted(df_impute["seed"].unique()):
            df_impute_iseed = df_impute[df_impute["seed"] == iseed].set_index("ID")
            param_dict = parse_filename_for_params(complete_data)
            results = evaluate_distributions(df_impute_iseed, gt, blocks, cols_to_eval, param_dict)
            results["normalisation"] = "none"
            results["seed"] = iseed
            df_results.append(results)

        df_results = pd.concat(df_results, ignore_index=False)
        df_cv = calculate_coverage_rate(df_impute, gt, blocks, cols_to_eval, param_dict, confidence_level=0.95)

        df_fs = compare_matrices(gt, df_impute, variables, algorithm, param_dict)

        quality_dir = os.path.join(os.path.dirname(imputed), 'quality')
        os.makedirs(quality_dir, exist_ok=True)

        df_results.to_csv(os.path.join(quality_dir, file_name + "_quality.csv"))
        df_cv.to_csv(os.path.join(quality_dir, file_name + "_coverage.csv"), index=False)
        df_fs.to_csv(os.path.join(quality_dir, file_name + "_similarities.csv"), index=False)
        
        print(f"Imputation quality metrics saved to {os.path.join(quality_dir, file_name + '_quality.csv')}")
        print(f"Imputation similarities metrics saved to {os.path.join(quality_dir, file_name + '_similarities.csv')}")

    elif algorithm == "Average":
        # imputed_df = pd.read_table(imputed, sep=',', index_col=0)
        df_impute = load_and_clean(imputed, variables, algorithm)
        cols_to_eval = sorted(        
        [col for col in blocks.columns if col not in ["Age", "Site"] and blocks[col].notna().any()],
        key=lambda x: int(x.split('_')[1]))

        df_results = []
        for iseed in sorted(df_impute["seed"].unique()):
            df_impute_iseed = df_impute[df_impute["seed"] == iseed].set_index("ID")
            param_dict = parse_filename_for_params(complete_data)
            results = evaluate_distributions(df_impute_iseed, gt, blocks, cols_to_eval, param_dict)
            results["normalisation"] = "none"
            results["seed"] = iseed
            df_results.append(results)
        
        df_results = pd.concat(df_results, ignore_index=False)
        df_cv = calculate_coverage_rate(df_impute, gt, blocks, cols_to_eval, param_dict, confidence_level=0.95)
        df_fs = compare_matrices(gt, df_impute, variables, algorithm, param_dict)

        quality_dir = os.path.join(os.path.dirname(imputed), 'quality')
        os.makedirs(quality_dir, exist_ok=True)
        
        df_results.to_csv(os.path.join(quality_dir, file_name + "_quality.csv"))
        # df_results.to_csv(os.path.join(output_dir, file_name + "_combined_results.csv"), index=False)
        df_cv.to_csv(os.path.join(quality_dir, file_name + "_coverage.csv"))
        df_fs.to_csv(os.path.join(quality_dir, file_name + "_similarities.csv"))
        
        print(f"Imputation quality metrics saved to {os.path.join(quality_dir, file_name + '_quality.csv')}")
        print(f"Imputation quality metrics saved to {os.path.join(quality_dir, file_name + '_similarities.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate quality of imputation")
    parser.add_argument('--algorithm', required=True, choices=["MICE", "ET", "AutoComplete", "Average"], help="Algorithm used for imputation")
    parser.add_argument('--complete_data', required=True, help="Path to the ground truth complete dataset")
    parser.add_argument('--blocks_data', required=True, help="Path to the block data")
    parser.add_argument('--global_norm_imputed', required=False, help="Path to the globally normalized imputed data")
    parser.add_argument('--strat_norm_imputed', required=False, help="Path to the stratified normalized imputed data")
    parser.add_argument('--imputed', required=False, help="Path to the imputed data if no normalization")
    parser.add_argument('--output_dir', required=True, help="Directory to save the results")
    args = parser.parse_args()

    main(args.algorithm, args.complete_data, args.blocks_data, args.global_norm_imputed, args.strat_norm_imputed, args.imputed, args.output_dir)
