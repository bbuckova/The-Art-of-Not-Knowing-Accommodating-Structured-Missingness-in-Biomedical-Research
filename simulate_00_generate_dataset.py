import numpy as np
import pandas as pd
import random
import os
import argparse

def generate_covariance_matrix(matrix_type, dim=10, rho=0.4):
    """
    Generate a covariance matrix and corresponding mean vector for simulation purposes.

    Parameters
    ----------
    matrix_type : str
        The type of covariance matrix to generate. Must be one of:
            - "independent": Identity matrix (no correlation between variables).
            - "fixed": Equicorrelated matrix with off-diagonal values set to `rho`.
            - "real": Empirical covariance and mean data loaded from file.
    
    dim : int, optional
        Dimensionality of the generated matrix (default is 10). 
        This sets the number of variables used from the loaded or constructed matrices.
    
    rho : float, optional
        Correlation coefficient used in the "fixed" matrix (default is 0.4).
        Ignored for other matrix types.

    Returns
    -------
    matrix : ndarray or DataFrame
        Covariance matrix of shape (dim+1, dim+1). The first variable is "Age".

    mean_df : ndarray or DataFrame
        Mean values for each variable (across groups in "real" mode) of shape (dim+1, n_groups) or (1, dim) for synthetic types.

    Notes
    -----
    - The "real" option loads precomputed matrices from:
        - `means.csv`: the empirical covariance matrix.
        - `cov_matrix.csv`: the empirical mean values.
    """
    if matrix_type == "independent":
        matrix = np.eye(dim)
        mean_df = np.zeros((1, dim))
    
    elif matrix_type == "fixed":
        matrix = np.full((dim, dim), rho)
        np.fill_diagonal(matrix, 1)
        mean_df = np.zeros((1, dim))

    elif matrix_type == "real":        
        matrix = pd.read_table("means.csv", sep=',',index_col=0)
        matrix = matrix.iloc[0:dim+1, 0:dim+1]
        mean_df = pd.read_table("cov_matrix.csv", sep=',', index_col=0)
        mean_df = mean_df.iloc[0:dim+1]
        
    else:
        raise ValueError(f"Unknown type: {matrix_type}. Choose from 'independent', 'fixed', or 'real'.")
    
    return matrix, mean_df

def generate_overlapping_vectors(n: int, overlap_size: int):
    """
    Creates two datasets of indices from 1 to n, with a specified overlap between them.
    
    Parameters:
    -----------
    n : int
        The total number of indices to be divided into two datasets.
    overlap_size : int
        The number of indices that will overlap between the two datasets.
        
    Returns:
    --------
    Tuple[List[int], List[int]]
        A tuple containing two lists of indices representing the two datasets.
        Each list has `overlap_size` indices in common, while the rest are unique.
        
    Raises:
    -------
    ValueError
        If the overlap_size is larger than n or if there are not enough indices
        to satisfy the overlap and unique indices requirements.
        
    """
    # Validate the parameters to ensure the operation is feasible
    if overlap_size > n:
        raise ValueError("Overlap size cannot exceed the total number of indices.")
    if (n - overlap_size) % 2 != 0:
        raise ValueError("Cannot evenly divide remaining indices between datasets. Adjust overlap size.")

    # Generate a list of indices from 1 to n
    indices = list(range(1, n + 1))
    
    # Randomly select the overlapping indices
    overlap_indices = random.sample(indices, overlap_size)
    
    # Identify the remaining indices after choosing the overlap
    remaining_indices = [i for i in indices if i not in overlap_indices]
    
    # Calculate the number of unique indices required for each dataset
    unique_size_per_dataset = (n - overlap_size) // 2
    
    # Randomly sample unique indices for Dataset 1
    unique_dataset1 = random.sample(remaining_indices, unique_size_per_dataset)
    
    # Remaining unique indices for Dataset 2
    unique_dataset2 = [i for i in remaining_indices if i not in unique_dataset1]
    
    # Combine overlap and unique indices for each dataset
    dataset1 = sorted(overlap_indices + unique_dataset1)
    dataset2 = sorted(overlap_indices + unique_dataset2)
    
    return [dataset1, dataset2]

def create_blocks(original_sample, vectors):
    """
    Introduce NaN values in `original_sample` based on the site-specific vectors.

    Parameters:
    -----------
    original_sample : pd.DataFrame
        DataFrame containing the sample data, including a 'site' column indicating dataset groups.
    vectors : list of lists
        A list of integer vectors where each vector corresponds to a site and contains 
        the indices of the columns to keep for that site.

    Returns:
    --------
    pd.DataFrame
        Modified DataFrame with NaN values introduced in columns not specified for each site.
    """
    
    # Get unique site identifiers and the maximum column index to work with
    sites = original_sample["Site"].unique()
    sites.sort()
    dim = max(max(sublist) for sublist in vectors)
    
    # Iterate through each site
    for isite in sites:
        to_remove = list(set(np.arange(1,dim+1)) - set(vectors[isite]))
        to_remove = original_sample.columns[to_remove]
        # original_sample[original_sample["site"] == isite, to_remove] = np.NaN
        original_sample.loc[original_sample["Site"] == isite, to_remove] = np.NaN


    return original_sample

def generate_dataset(matrix_type, dim=10, n_samples=50, dataset=0):
    """
    Generates a dataset with one variable sampled from a uniform distribution (e.g., Age) 
    and the remaining variables sampled from a conditional multivariate normal distribution.
    
    Parameters:
    - matrix_type: Type of covariance matrix to generate (to be defined in generate_covariance_matrix).
    - dim: Dimension of the multivariate normal distribution (excluding the uniform variable).
    - n_samples: Number of data points (rows) to generate.
    - dataset: number of dataset to be generated
    
    Returns:
    - DataFrame containing the generated dataset with 'Age' and remaining multivariate normal variables.
    """
    
    # Generate the covariance matrix and mean vector
    cov_m, mean_df = generate_covariance_matrix(matrix_type, dim=dim, is_server=is_server)
    cov_m_np = cov_m.to_numpy()  # Convert to NumPy array
    mean_df = mean_df.to_numpy()  # Convert to NumPy array

    # Generate known uniform values (e.g., Age)
    known_uniform_values = np.array([random.uniform(18, 60) for _ in range(n_samples)])
    known_indices = np.array([0])  # Index for 'Age' (uniform distribution)

    # Identify unknown indices (multivariate normal variables)
    all_indices = np.arange(dim+1)
    unknown_indices = np.delete(all_indices, known_indices)

    # Partition the mean vector and covariance matrix
    mu_known = mean_df[known_indices, dataset]  # Mean vector for known variables (Age)
    mu_unknown = mean_df[unknown_indices, dataset]  # Mean vector for unknown variables
    mu_unknown = np.random.choice([-1, 1], size=dim) * mu_unknown * 0.1

    cov_known_known = cov_m_np[np.ix_(known_indices, known_indices)]
    cov_unknown_unknown = cov_m_np[np.ix_(unknown_indices, unknown_indices)]
    cov_unknown_known = cov_m_np[np.ix_(unknown_indices, known_indices)]
    cov_known_unknown = cov_m_np[np.ix_(known_indices, unknown_indices)]

    # Compute conditional mean and covariance
    cov_inv_known_known = np.linalg.inv(cov_known_known)
    conditional_mean = mu_unknown + (cov_unknown_known @ (cov_inv_known_known @ ((known_uniform_values - mu_known)[np.newaxis,:]))).T
    conditional_cov = cov_unknown_unknown - cov_unknown_known @ cov_inv_known_known @ cov_known_unknown

    # Sample from the conditional multivariate normal distribution
    dependent_data = np.array([np.random.multivariate_normal(c_mu, conditional_cov, 1)[0] for c_mu in conditional_mean])

    # Combine the uniform values (Age) with the generated normal data
    full_data = pd.DataFrame(np.column_stack((known_uniform_values, dependent_data)), 
                                columns=["Age"] + [f"Variable_{i+1}" for i in range(dim)])
    
    return full_data

def generate_samples_with_missing_values(filename, destination_dir, num_samples=500, matrix_type="real", overlap=3, 
                                         num_datasets=2, vector_length=8, noise=0.1, seed=42, dim=40, missingness="noise-blocks"):
    np.random.seed(int(seed))
    # Choose how to define dimensions and overlaps
    if dim == 0:
        dim = num_datasets * (vector_length - overlap) + overlap

    # Generate Sample
    # original_sample, cov_matrix, mean_vectors = sample_multivariate_normal(matrix_type, dim=dim, num_samples=num_samples)
    f1 = generate_dataset(matrix_type, dim=dim, n_samples=num_samples, dataset=0)
    f1["Site"] = 0
    f2 = generate_dataset(matrix_type, dim=dim, n_samples=num_samples, dataset=1)
    f2["Site"] = 1

    original_sample = pd.concat([f1,f2])
    original_sample = original_sample.reset_index(drop=1)
    original_sample.index.name = "ID"
    to_process = original_sample.copy()

    # Generate Subsets
    vectors = generate_overlapping_vectors(dim, overlap)

    # Introduce missingness
    if "noise" in missingness:
        # Introducing missing
        df_noise =pd.DataFrame((np.random.uniform(0, 1, 
                            (num_samples*num_datasets, dim)) >= (1-noise)).astype(int), 
                            index=original_sample.index, 
                            columns=list(original_sample.columns[1:dim+1]))#list(set(original_sample.columns)-set(["Age","Site"])))

        sample_noise = original_sample[list(original_sample.columns[1:dim+1])].mask(df_noise == 1)
        sample_noise["Site"] = original_sample["Site"]
        sample_noise.insert(0, "Age", original_sample["Age"])
        to_process = sample_noise.copy()

    if "blocks" in missingness:
        if "Age" not in to_process.columns:
            to_process["Site"] = original_sample["Site"]
            to_process.insert(0, "Age", original_sample["Age"])
        
        to_process = create_blocks(to_process, vectors)

    if "dependence" in missingness: 
        q=noise
        dependence_df = original_sample.copy()
        vectors_overlapping = list(filter(lambda x: x in vectors[0], vectors[1]))
        A_standardized = (original_sample.iloc[:,vectors_overlapping[0]] - original_sample.iloc[:,vectors_overlapping[0]].mean()) / original_sample.iloc[:,vectors_overlapping[0]].std()
        # prob_nan = np.exp(3 * abs(A_standardized)) / (1 + np.exp(3 * abs(A_standardized)))
        prob_nan = np.exp(2*A_standardized) / (1 + np.exp(2*A_standardized))
        dependence_df.iloc[:,vectors_overlapping[1]] = dependence_df.iloc[:,vectors_overlapping[1]].mask(np.random.rand(len(dependence_df))>prob_nan)
        dependence_df.iloc[:,vectors_overlapping[2]] = dependence_df.iloc[:,vectors_overlapping[2]].mask(dependence_df.iloc[:,vectors_overlapping[1]].isna() & (np.random.rand(len(dependence_df)) < q))
        to_process = to_process.mask(dependence_df.isna())

    to_process.to_csv(os.path.join(destination_dir, f"{filename}_blocks.csv"), sep=",")
    original_sample.to_csv(os.path.join(destination_dir, f"{filename}_complete.csv"), sep=",")
    

def main():
    allowed_missings = ['noise', 'blocks', "dependence", 'noise-blocks', "noise-dependence", "blocks-dependence", "noise-blocks-dependence"]
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate samples with missing values and structured blocks.")

    # Add arguments for the function parameters
    parser.add_argument("filename", type=str, help="Base filename for output files")
    parser.add_argument("destination_dir", type=str, help="Directory where output files will be saved")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--matrix_type", type=str, default="real", choices=["real", "fixed", "independent"], help="Covariance matrix type")
    parser.add_argument("--overlap", type=int, default=3, help="Number of overlapping elements between datasets")
    parser.add_argument("--num_datasets", type=int, default=2, help="Number of datasets to generate")
    parser.add_argument("--vector_length", type=int, default=8, help="Length of each vector for the datasets")
    parser.add_argument("--noise", type=float, default=0.1, help="Proportion of noise to introduce in the samples")
    parser.add_argument("--dim", type=float, default=40, help="Number of vectors in the dataset do be generated")
    parser.add_argument("--missingness", type=str, choices=allowed_missings, default="noise_blocks", help="Number of vectors in the dataset do be generated")
    parser.add_argument("--seed", type=float, default=42, help="Set seed for reproducibility")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    generate_samples_with_missing_values(
        filename=args.filename,
        destination_dir=args.destination_dir,
        num_samples=args.num_samples,
        matrix_type=args.matrix_type,
        overlap=args.overlap,
        num_datasets=args.num_datasets,
        vector_length=args.vector_length,
        noise=args.noise,
        dim = args.dim,
        missingness = args.missingness,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()