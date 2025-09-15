# The Art of Not Knowing: Accommodating Structured Missingness in Biomedical Research

This repository contains scripts to a full pipeline for generating, normalizing, imputing, and evaluating simulated datasets with structured missingness. It supports multiple imputation methods including AutoComplete, Extra Trees (ET), and MICE (standard and context-based). The pipeline is modular and designed to run locally or on an HPC cluster using SLURM.

---

## 📦 Environment Requirements

This pipeline assumes the following environments and tools are available:

### Conda Environments:
- `ML`: Python-based environment containing:
  - Python ≥ 3.8
  - `numpy`, `pandas`, `scikit-learn`
  - `pyarrow` (for MICE output)
- `R_env`: R-based environment with:
  - R ≥ 4.0
  - `mice`, `arrow`, and relevant tidyverse packages

### Other Dependencies:
- [AutoComplete](https://github.com/sriramlab/AutoComplete) package has to be downloaded and set-up
- SLURM job scheduler (for cluster execution)
- Scripts referenced in `code_dir`:
  - `simulate_00_generate_dataset.py`
  - `simulate_01_normalise.py`
  - `simulate_02_AutoComplete.sh`
  - `simulate_02_ExtraTrees.py`
  - `simulate_02_MICE.R`
  - `simulate_02_MICE_context.R`
  - `simulate_03_eval.py`
  - `utils_concat_csv.py`
---

## 🧠 Functional Overview

### 1. **Data Simulation**
- **Function:** `simulate_data`
- **Description:** Generates synthetic datasets using various parameters (samples, overlap, noise, missingness type).
- **Script:** `simulate_00_generate_dataset.py`

### 2. **Data Normalization**
- **Function:** `normalize_data`
- **Description:** Normalizes simulated datasets for consistent preprocessing.
- **Script:** `simulate_01_normalise.py`

### 3. **AutoComplete Imputation**
- **Function:** `Autocomplete_run`
- **Description:** Applies the AutoComplete model to impute missing data.
- **Script:** `simulate_02_AutoComplete.sh`

### 4. **AutoComplete Evaluation**
- **Function:** `Autocomplete_evaluate`
- **Description:** Evaluates AutoComplete imputations against ground truth.
- **Script:** `simulate_03_eval.py`

### 5. **Extra Trees Imputation**
- **Function:** `ET_run`
- **Description:** Uses ExtraTreesRegressor to impute missing data. Can run locally or via SLURM.
- **Script:** `simulate_02_ExtraTrees.py`

### 6. **Extra Trees Evaluation**
- **Function:** `ET_evaluate`
- **Description:** Evaluates imputed data from Extra Trees.
- **Script:** `simulate_03_eval.py`

### 7. **MICE Standard Imputation**
- **Function:** `MICE_standard_run`
- **Description:** Applies standard MICE (Multiple Imputation by Chained Equations).
- **Script:** `simulate_02_MICE.R`

### 8. **MICE Contextual Imputation**
- **Function:** `MICE_context_run`
- **Description:** Applies a context-aware variant of MICE.
- **Script:** `simulate_02_MICE_context.R`

### 9. **MICE Evaluation**
- **Function:** `MICE_evaluate`
- **Description:** Evaluates MICE imputations.
- **Script:** `simulate_03_eval.py`

---

## 🛠️ Usage

```bash
./simulate.sh run_simulations /where/to/save/resluts
./simulate.sh normalize_data /where/to/save/resluts
./simulate.sh Autocomplete_run /where/to/save/resluts 0 1 /path/to/AutoComplete
./simulate.sh Autocomplete_evaluate /where/to/save/resluts 0 1 
./simulate.sh ET_run /where/to/save/resluts 0 1 
./simulate.sh ET_evaluate /where/to/save/resluts 0 1 
./simulate.sh MICE_standard_run /where/to/save/resluts 0 1 /path/to/MICE_libs
./simulate.sh MICE_context_run /where/to/save/resluts 0 1 /path/to/MICE_libs
./simulate.sh MICE_evaluate /where/to/save/resluts 0 1
```
> Replace `0` (run locally) with `1` to submit jobs to SLURM where supported.  
> Ensure you're running the scripts from the repository folder.

---

## 📁 File Naming and Structure

Generated files follow a naming convention like:

```
samples1000_overlap0.5_noise0.1_noise-blocks.csv
samples1000_overlap0.5_noise0.1_noise-blocks_norm.csv
samples1000_overlap0.5_noise0.1_noise-blocks_blocks.csv
samples1000_overlap0.5_noise0.1_noise-blocks_complete.csv
```

Output folders:
- `normalized/`
- `AC_results/`
- `ET_results/`
- `MICE_standard_results/`
- `MICE_context_results/`

---
## License

This project uses dual licensing:
- **Code**: GPL-3.0 (see `LICENSE`)
- **Data files** (`cov_matrix.csv`, `means.csv`): CC0-1.0 (see `LICENSE-CC0`)

For complete licensing information, see `LICENSES.md`.

