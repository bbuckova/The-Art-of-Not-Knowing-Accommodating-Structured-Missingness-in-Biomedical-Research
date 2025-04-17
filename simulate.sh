#!/bin/bash
source activate ML

# Default destination/data directory and overwrite flag
# default_destination_dir="simulations"
default_overwrite=0

# Function to simulate data using Python script
simulate_data() {
  # Check if the destination directory exists; create it if not
  if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
    echo "Directory created: $destination_dir"
  fi

  # Parse inputs
  local is_server=$1
  local overwrite=$2
  local sample=$3
  local overlap=$4
  local noise=$5
  local missingness=$6
  local destination_dir=${7:-$default_destination_dir}  # Use default if not provided

  # Create a unique filename for each run
  local filename="${filename_base}_samples${sample}_overlap${overlap}_noise${noise}_${missingness}"

  # Full path to the output file
  local output_file="${destination_dir}/${filename}.csv"

  # Check if file exists and overwrite is set to 0
  if [ -f "$output_file" ] && [ "$overwrite" -eq 0 ]; then
    echo "File $output_file already exists. Skipping generation."
    return
  fi

  # Run the Python script with the current parameter combination
  python "$code_dir/simulate_00_agedep.py" \
    "$filename" "$destination_dir" \
    --num_samples "$sample" \
    --matrix_type "$matrix_type" \
    --overlap "$overlap" \
    --num_datasets "$num_datasets" \
    --vector_length "$vector_length" \
    --noise "$noise" \
    --seed "$seed" \
    --missingness "$missingness" \
    --is_server "$is_server"

  # Print status message
  echo "Generated samples with num_samples=$sample, overlap=$overlap, noise=$noise, missingness=$missingness"
}

simulate_diverse_data() {
  # Check if the destination directory exists; create it if not
  if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
    echo "Directory created: $destination_dir"
  fi

  # Parse inputs
  local is_server=$1
  local overwrite=$2
  local sample=$3
  local overlap=$4
  local noise=$5
  local missingness=$6
  local destination_dir=${7:-$default_destination_dir}  # Use default if not provided

  # Create a unique filename for each run
  local filename="${filename_base}_samples${sample}_overlap${overlap}_noise${noise}_${missingness}"

  # Full path to the output file
  local output_file="${destination_dir}/${filename}.csv"

  # Check if file exists and overwrite is set to 0
  if [ -f "$output_file" ] && [ "$overwrite" -eq 0 ]; then
    echo "File $output_file already exists. Skipping generation."
    return
  fi

  # Run the Python script with the current parameter combination
  python "$code_dir/simulate_00_agedep_cohorts.py" \
    "$filename" "$destination_dir" \
    --num_samples "$sample" \
    --matrix_type "$matrix_type" \
    --overlap "$overlap" \
    --num_datasets "$num_datasets" \
    --vector_length "$vector_length" \
    --noise "$noise" \
    --seed "$seed" \
    --missingness "$missingness" \
    --is_server "$is_server"

  # Print status message
  echo "Generated samples with num_samples=$sample, overlap=$overlap, noise=$noise, missingness=$missingness"
}

# Function to normalize data using another Python script
normalize_data() {
  local destination_dir=${1:-$default_destination_dir}  # Use default if not provided
  local overwrite=${2:-$default_overwrite}              # Use default if not provided

  # Run the normalization script with arguments
  python "$code_dir/simulate_01_normalise.py" "$destination_dir" --overwrite $overwrite

  # Print status message
  echo "Normalization complete for data in directory $destination_dir"
}

# Function to loop through parameter combinations and run simulations
run_simulations() {
  local destination_dir=${1:-$default_destination_dir}  # Use default if not provided
  
  # Define the types of missingness
  local missingness_types=("noise" "blocks" "dependence" "noise-blocks" "noise-dependence" "noise-blocks-dependence")

  for sample in "${samples[@]}"; do
    for overlap in "${overlaps[@]}"; do
      for noise in "${noises[@]}"; do
        for missingness in "${missingness_types[@]}"; do
          simulate_data "$run_at_server" "$overwrite" "$sample" "$overlap" "$noise" "$missingness" "$destination_dir"
        done
      done
    done
  done
}

run_simulations_cohorts() {
  local destination_dir=${1:-$default_destination_dir}  # Use default if not provided
  
  # Define the types of missingness
  local missingness_types=("noise" "blocks" "dependence" "noise-blocks" "noise-dependence" "noise-blocks-dependence")

  for sample in "${samples[@]}"; do
    for overlap in "${overlaps[@]}"; do
      for noise in "${noises[@]}"; do
        for missingness in "${missingness_types[@]}"; do
          simulate_diverse_data "$run_at_server" "$overwrite" "$sample" "$overlap" "$noise" "$missingness" "$destination_dir"
        done
      done
    done
  done
}

# Function to run the AutoComplete process on normalized data
run_AutoComplete() {  
  local lr=0.1
  local batch_size=25
  local epochs=1000
  local momentum=0.9
  local encoding_ratio=1
  local copymask_amount=0.3
  local simulate_missing=0.01
  local depth=1

  cd $1/normalized

  # Find stratified normalized files and run the AutoComplete process
  find . -maxdepth 1 -type f -name "*_norm.csv" | while read -r ifile 
  do
    Data_to_analyze="$(realpath "$ifile")"
    bash "$code_dir/simulate_02_AutoComplete.sh" \
      "$2" \
      "$Data_to_analyze" \
      "$lr" \
      "$batch_size" \
      "$epochs" \
      "$momentum" \
      "$encoding_ratio" \
      "$copymask_amount" \
      "$simulate_missing" \
      "$depth" \
      "$3"
  done
}

evaluate_AutoComplete() {    
  # Define the code directory path
  code_dir="/project/3022000.05/precognition/scripts/Precognition_scripts"
  
  # Check if the entire function should be submitted as a single cluster job
#   if [ "$2" == "1" ]; then
#     echo "Submitting the entire function to the cluster as one job."

#     # Submit to the cluster with Slurm, changing $2 to "2" to prevent resubmission
#     sbatch <<EOF
# #!/bin/bash
# #SBATCH --job-name=AutoCompleteEvaluation
# #SBATCH --output=${1}/AC_results/cluster_job_%j.out
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=1   
# #SBATCH --mem=5G
# #SBATCH --time=2:00:00

# module load python

# source activate ML
# $(declare -f evaluate_AutoComplete)
# evaluate_AutoComplete "$1" "2" "$3" "$4" # Call the function with modified "$2"
# EOF
#     return  # Exit after submission
#   fi

#   # Skip re-submission if running in the cluster (i.e., $2 is now "2")
#   if [ "$2" == "2" ]; then
#     echo "Already running on the cluster. Skipping re-submission."
#   fi

# Process each seed directory within AC_results
  SIM_PATH="${1}/AC_results" 
  if [ ! -d $SIM_PATH/concatenated ]; then 
      python utils_concat_csv.py --parent_dir $SIM_PATH --method AC
  fi
  cd "$SIM_PATH" || { echo "Failed to change directory to $SIM_PATH"; exit 1; }

  # Find stratified normalized files and run the evaluation process
  find "$1" -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
    # Extract COMPLETE_DATA and prepare BLOCKS_DATA and CORENAME
    COMPLETE_DATA="$(realpath "$ifile")"
    BLOCKS_DATA="${COMPLETE_DATA/_complete.csv/_blocks.csv}"
    
    FILENAME=$(basename "$COMPLETE_DATA")
    CORENAME="${FILENAME/_complete.csv/}"

    # Define paths for imputed files within the current seed directory, using the extracted seed
    GLOBAL_NORM_IMPUTED="${SIM_PATH}/concatenated/${CORENAME}_global_norm_imputed_concatenated.csv"
    STRAT_NORM_IMPUTED="${SIM_PATH}/concatenated/${CORENAME}_strat_norm_imputed_concatenated.csv"
    FINAL_FILE="${SIM_PATH}/quality/${CORENAME}_quality.csv"

    # Check if FINAL_FILE exists and overwrite is set to 0
    if [ "$3" == "0" ] && [ -f "$FINAL_FILE" ]; then
      echo "File ${FINAL_FILE} already exists and overwrite is set to 0. Skipping evaluation for seed $seed."
      continue
    fi

    # Run Python script for AutoComplete evaluation
    python "${4}/simulate_03_eval.py" \
      --algorithm AutoComplete \
      --complete_data "$COMPLETE_DATA" \
      --blocks_data "$BLOCKS_DATA" \
      --global_norm_imputed "$GLOBAL_NORM_IMPUTED" \
      --strat_norm_imputed "$STRAT_NORM_IMPUTED" \
      --output_dir "$SIM_PATH"
  done
}

run_ET() {  
  cd "$1" || { echo "Failed to change directory to $1"; exit 1; }

  # Define the base directory for target folders based on the server flag
    Results=$SIM_PATH/"ET_results"
  
  # Ensure the Results directory exists
  mkdir -p "$Results"

  # Loop through seeds 1 to 10
  for seed in {1..10}; do #{1..10}
    # Define target directory for the current seed
    TARGET_DIR="${Results}/seed_${seed}"
    mkdir -p "$TARGET_DIR"

    # Find stratified normalized files and run the AutoComplete process
    find . -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
      Data_to_analyze="$(realpath "$ifile")"
      Data_missing="${Data_to_analyze/_complete.csv/_blocks.csv}"
      Data_results="${TARGET_DIR}/$(basename "${Data_to_analyze/_complete.csv/_imputed.csv}")"

      # Check if Data_results exists in TARGET_DIR and OVERWRITE is set to 0
      if [ "$3" == "0" ] && [ -e "$Data_results" ]; then
        echo "Data_results $Data_results already exists and OVERWRITE is set to 0. Skipping..."
        continue  # Exit the current iteration of the for-loop for this seed
      fi

      # Check if running on server or locally, then initiate Python script
      if [ "$2" == "1" ]; then
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ET_impute_seed_$seed
#SBATCH --output="${TARGET_DIR}/impute_seed_${seed}.out"
#SBATCH --error="${TARGET_DIR}/impute_seed_${seed}.err"
#SBATCH --time=05:00:00  
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --mem=12G

source activate ML
python "$code_dir/simulate_02_ExtraTrees.py" "$TARGET_DIR" "$Data_to_analyze" "$Data_missing" --seed "$seed"
EOF
      else
        python "$code_dir/simulate_02_ExtraTrees.py" "$TARGET_DIR" "$Data_to_analyze" "$Data_missing" --seed "$seed"
      fi

      # Print a message to indicate completion for the current seed
      echo "Imputation initiated for seed $seed and will save to $TARGET_DIR"
    done
  done
}


evaluate_ET() {    
#  
    # Set the current seed directory as SIM_PATH
    SIM_PATH="${1}/ET_results"

    # Check if the concatenated directory exists and run concatenate_csvs if necessary
    if [ ! -d "$SIM_PATH/concatenated" ]; then 
        python utils_concat_csv.py --parent_dir $SIM_PATH --method ET
    fi
    
    SIM_PATH="${1}/ET_results/concatenated"
    cd "$SIM_PATH" || { echo "Failed to change directory to $SIM_PATH"; exit 1; }

    find "$1" -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
      # Extract COMPLETE_DATA, BLOCKS_DATA, and CORENAME
      COMPLETE_DATA="$(realpath "$ifile")"
      BLOCKS_DATA="${COMPLETE_DATA/_complete.csv/_blocks.csv}"

      FILENAME=$(basename "$COMPLETE_DATA")
      CORENAME="${FILENAME/_complete.csv/}"

      # Define path for imputed file within the current seed directory
      IMPUTED="${SIM_PATH}/${CORENAME}_concatenated.csv"
      QUALITY="${SIM_PATH}/quality/${CORENAME}_quality.csv"

      # Check for overwrite condition
      if [ "$3" == "0" ] && [ -f "$QUALITY" ]; then
          echo "Imputed file $IMPUTED already exists and overwrite is set to 0. Skipping..."
          continue  # Skip this iteration without exiting the loop
      fi

      # Run Python script for ExtraTrees evaluation
      python "$code_dir/simulate_03_eval.py" \
        --algorithm ET \
        --complete_data "$COMPLETE_DATA" \
        --blocks_data "$BLOCKS_DATA" \
        --imputed "$IMPUTED" \
        --output_dir "$SIM_PATH"

      echo "Evaluation completed for $IMPUTED"
      
      done
}


run_MICE_standard() {  
  cd "$1" || { echo "Failed to change directory to $1"; exit 1; }

  # Define the base directory for target folders based on the server flag
  Results=$SIM_PATH/"ET_results"
  
  # Ensure the Results directory exists
  mkdir -p "$Results"

  # Find stratified normalized files and run the AutoComplete process
  find . -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
    Data_to_analyze="$(realpath "$ifile")"
    Data_missing="${Data_to_analyze/_complete.csv/_blocks.csv}"
    Data_results="${TARGET_DIR}/$(basename "${Data_to_analyze/_complete.csv/_imputed.csv}")"

    # Check if Data_results exists in TARGET_DIR and OVERWRITE is set to 0
    if [ "$3" == "0" ] && [ -e "$Data_results" ]; then
      echo "Data_results $Data_results already exists and OVERWRITE is set to 0. Skipping..."
      continue  # Exit the current iteration of the for-loop for this seed
    fi

    # Check if running on server or locally, then initiate Python script
    if [ "$2" == "1" ]; then
      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=MICE_impute
#SBATCH --output="${Results}/impute_seed.out"
#SBATCH --error="${Results}/impute_seed.err"
#SBATCH --time=01:00:00  
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G

source activate R_env
module unload R/4.3.3

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
export R_LIBS_USER=/home/preclineu/barbuc/.conda/envs/R_env

Rscript "$code_dir/simulate_02_MICE.R" "$Data_missing" "$Results"
EOF
      else
        Rscript "$code_dir/simulate_02_MICE.R" "$Data_missing" "$Results"
      fi

    done
}

run_MICE_context() {  
  cd "$1" || { echo "Failed to change directory to $1"; exit 1; }

  # Define the base directory for target folders based on the server flag
  Results=$SIM_PATH/"ET_results"
  
  # Ensure the Results directory exists
  mkdir -p "$Results"

  # Find stratified normalized files and run the AutoComplete process
  find . -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
    Data_to_analyze="$(realpath "$ifile")"
    Data_missing="${Data_to_analyze/_complete.csv/_blocks.csv}"
    Data_results="${Results}/$(basename "${Data_to_analyze/_complete.csv/_imputed.parquet}")"

    # Check if Data_results exists in TARGET_DIR and OVERWRITE is set to 0
    if [ "$3" == "0" ] && [ -e "$Data_results" ]; then
      echo "Data_results $Data_results already exists and OVERWRITE is set to 0. Skipping..."
      continue  # Exit the current iteration of the for-loop for this seed
    fi

    # Check if running on server or locally, then initiate Python script
    if [ "$2" == "1" ]; then
      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=MICE_context
#SBATCH --output="${Results}/impute_seed.out"
#SBATCH --error="${Results}/impute_seed.err"
#SBATCH --time=04:00:00  
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --mem=8G

source activate R_env
module unload R/4.3.3

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
export R_LIBS_USER=/home/preclineu/barbuc/.conda/envs/R_env

Rscript "$code_dir/simulate_02_MICE_context.R" "$Data_missing" "$Results"
EOF
      else
        Rscript "$code_dir/simulate_02_MICE_context.R" "$Data_missing" "$Results"
      fi

    done
}

run_MICE_hnorm() {  
  cd "$1" || { echo "Failed to change directory to $1"; exit 1; }

  # Define the base directory for target folders based on the server flag
  Results=$SIM_PATH/"ET_results"
  
  # Ensure the Results directory exists
  mkdir -p "$Results"

  # Find stratified normalized files and run the AutoComplete process
  find . -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
    Data_to_analyze="$(realpath "$ifile")"
    Data_missing="${Data_to_analyze/_complete.csv/_blocks.csv}"
    Data_results="${Results}/$(basename "${Data_to_analyze/_complete.csv/_imputed.parquet}")"

    # Check if Data_results exists in TARGET_DIR and OVERWRITE is set to 0
    if [ "$3" == "0" ] && [ -e "$Data_results" ]; then
      echo "Data_results $Data_results already exists and OVERWRITE is set to 0. Skipping..."
      continue  # Exit the current iteration of the for-loop for this seed
    fi

    # Check if running on server or locally, then initiate Python script
    if [ "$2" == "1" ]; then
      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=MICE_hnorm
#SBATCH --output="${Results}/impute_seed.out"
#SBATCH --error="${Results}/impute_seed.err"
#SBATCH --time=01:00:00  
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G

source activate R_env
module unload R/4.3.3

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8
export R_LIBS_USER=/home/preclineu/barbuc/.conda/envs/R_env

Rscript "$code_dir/simulate_02_hnorm_MICE.R" "$Data_missing" "$Results"
EOF
      else
        Rscript "$code_dir/simulate_02_hnorm_MICE.R" "$Data_missing" "$Results"
      fi

    done
}


evaluate_MICE() {
  # Check if the entire function should be submitted as a single cluster job
  # Process files locally if not submitting to the cluster
  find "$1" -maxdepth 1 -type d -name "MICE*" | while read -r mice_dir; do
    echo "Processing MICE directory: $mice_dir"

      # Set the current seed directory as SIM_PATH
      SIM_PATH=${mice_dir}
      cd "$SIM_PATH" || { echo "Failed to change directory to $SIM_PATH"; exit 1; }

      # Process each file in this seed directory
      find "$1" -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
        # Extract COMPLETE_DATA and prepare BLOCKS_DATA and CORENAME
        COMPLETE_DATA="$(realpath "$ifile")"
        BLOCKS_DATA="${COMPLETE_DATA/_complete.csv/_blocks.csv}"
        
        FILENAME=$(basename "$COMPLETE_DATA")
        CORENAME="${FILENAME/_complete.csv/}"

        # Define path for imputed file within the current seed directory, using the extracted seed
        IMPUTED="${SIM_PATH}/${CORENAME}_imputed.parquet"
        QUALITY="${SIM_PATH}/quality/${CORENAME}_quality.csv"

        # Check for overwrite condition
        if [ "$3" == "0" ] && [ -f "$QUALITY" ]; then
          echo "Imputed file $IMPUTED already exists and overwrite is set to 0. Skipping..."
          continue  # Skip to the next ifile without exiting the seed_dir loop
        fi

        # Run Python script for MICE evaluation
        python "$code_dir/simulate_03_eval.py" \
          --algorithm MICE \
          --complete_data "$COMPLETE_DATA" \
          --blocks_data "$BLOCKS_DATA" \
          --imputed "$IMPUTED" \
          --output_dir "$SIM_PATH"
        
        echo "Evaluation completed for $IMPUTED"
      done
    done
}

run_average() {  
  cd "$1" || { echo "Failed to change directory to $1"; exit 1; }

  # Define the base directory for target folders based on the server flag
  Results=$SIM_PATH/"ET_results"
  
  # Ensure the Results directory exists
  mkdir -p "$Results"

  # Although average will always be the same, we'll run it multiple times as it simplifies the evaluations and runs quickly
  for seed in {1..10}; do
    # Define target directory for the current seed
    TARGET_DIR="${Results}/seed_${seed}"
    mkdir -p "$TARGET_DIR"

    # Find stratified normalized files and run the AutoComplete process
    find . -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
      Data_to_analyze="$(realpath "$ifile")"
      Data_missing="${Data_to_analyze/_complete.csv/_blocks.csv}"
      Data_results="${TARGET_DIR}/$(basename "${Data_to_analyze/_complete.csv/_imputed.csv}")"

      # Check if Data_results exists in TARGET_DIR and OVERWRITE is set to 0
      if [ "$3" == "0" ] && [ -e "$Data_results" ]; then
        echo "Data_results $Data_results already exists and OVERWRITE is set to 0. Skipping..."
        continue  # Exit the current iteration of the for-loop for this seed
      fi

      # Check if running on server or locally, then initiate Python script
      if [ "$2" == "1" ]; then
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=Average_impute_seed_$seed
#SBATCH --output="${TARGET_DIR}/impute_seed_${seed}.out"
#SBATCH --error="${TARGET_DIR}/impute_seed_${seed}.err"
#SBATCH --time=00:30:00  
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=12G

source activate ML
python "$code_dir/simulate_02_Average.py" "$TARGET_DIR" "$Data_to_analyze" "$Data_missing" --seed "$seed"
EOF
        else
          python "$code_dir/simulate_02_Average.py" "$TARGET_DIR" "$Data_to_analyze" "$Data_missing" --seed "$seed"
        fi

        # Print a message to indicate completion for the current seed
        echo "Imputation initiated for seed $seed and will save to $TARGET_DIR"
    done
  done
}

evaluate_average() {    
#  
    # Set the current seed directory as SIM_PATH
    SIM_PATH="${1}/Average_results"

    # Check if the concatenated directory exists and run concatenate_csvs if necessary
    if [ ! -d "$SIM_PATH/concatenated" ]; then 
        python utils_concat_csv.py --parent_dir $SIM_PATH --method ET
    fi
    
    SIM_PATH="${1}/Average_results/concatenated"
    cd "$SIM_PATH" || { echo "Failed to change directory to $SIM_PATH"; exit 1; }

    find "$1" -maxdepth 1 -type f -name "*_complete.csv" | while read -r ifile; do
      # Extract COMPLETE_DATA, BLOCKS_DATA, and CORENAME
      COMPLETE_DATA="$(realpath "$ifile")"
      BLOCKS_DATA="${COMPLETE_DATA/_complete.csv/_blocks.csv}"

      FILENAME=$(basename "$COMPLETE_DATA")
      CORENAME="${FILENAME/_complete.csv/}"

      # Define path for imputed file within the current seed directory
      IMPUTED="${SIM_PATH}/${CORENAME}_concatenated.csv"
      QUALITY="${SIM_PATH}/quality/${CORENAME}_quality.csv"

      # Check for overwrite condition
      if [ "$3" == "0" ] && [ -f "$QUALITY" ]; then
          echo "Imputed file $IMPUTED already exists and overwrite is set to 0. Skipping..."
          continue  # Skip this iteration without exiting the loop
      fi

      # Run Python script for ExtraTrees evaluation
      python "$code_dir/simulate_03_eval.py" \
        --algorithm Average \
        --complete_data "$COMPLETE_DATA" \
        --blocks_data "$BLOCKS_DATA" \
        --imputed "$IMPUTED" \
        --output_dir "$SIM_PATH"

      echo "Evaluation completed for $IMPUTED"
      
      done
}





# Display help message
usage() {
  echo "Usage: $0 {all|simulate|normalize|autocomplete|evaluate|run_ET} [destination_dir] [overwrite] [run_at_server]"
  echo "Options:"
  echo "  simulate                Run only the simulations"
  echo "  normalize               Run only the normalization"
  echo "  autocomplete            Run AutoComplete analysis on normalized data"
  echo "  autocomplete_evaluate   Evaluate AutoComplete results"
  echo "  ET_run                  Run Extra Trees imputation"
  echo "  ET_evaluate             Evaluate Extra Trees imputation"
  echo "  MICE_standard_run       Run MICE imputation"
  echo "  MICE_hnorm_run          Run MICE imputation"
  echo "  MICE_context_run        Run MICE imputation"
  echo "  MICE_evaluate           Evaluate MICE imputation"
  echo "  Average_run             Run average imputation"
  echo "  Average_evaluate        Evaluate average imputation"
  echo "Arguments:"
  echo "  destination_dir (optional)  Directory to save generated and normalized files"
  echo "  overwrite       (optional)  Set to 1 to overwrite existing normalized files"
  echo "  run_at_server   (optional)  Set to 1 to change paths for server execution"
}

# Main script logic
if [ $# -eq 0 ]; then
  usage
  exit 1
fi

# Parse arguments for optional parameters
destination_dir=${2:-$default_destination_dir}  # If not provided, use default
overwrite=${3:-$default_overwrite}              # If not provided, use default
run_at_server=${4:-0}                           # Default to 0 if not provided

# Define arrays for the varying parameters
samples=(50 500 5000)
overlaps=(4 10 20)
noises=(0.1 0.3 0.5 0.7)

# Set other fixed parameters
filename_base="simulation_"
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
matrix_type="real"
num_datasets=2
vector_length=8
seed=42

# Execute based on the first argument
case "$1" in
  simulate)
    run_simulations "$destination_dir"
    ;;
  simulate_site_effects)
    run_simulations_cohorts "$destination_dir"
    ;;
  normalize)
    normalize_data "$destination_dir" "$overwrite"
    ;;
  autocomplete)
    run_AutoComplete "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  autocomplete_evaluate)
    evaluate_AutoComplete "$destination_dir" "$run_at_server" "$overwrite"  "$code_dir"
    ;;
  ET_run)
    run_ET "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  ET_evaluate)
    evaluate_ET "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  MICE_standard_run)
    run_MICE_standard "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  MICE_hnorm_run)
    run_MICE_hnorm "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  MICE_context_run)
    run_MICE_context "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  MICE_evaluate)
    evaluate_MICE "$destination_dir" "$run_at_server" "$overwrite"
  ;;
  Average_run)
    run_average "$destination_dir" "$run_at_server" "$overwrite"
    ;;
  Average_evaluate)
    evaluate_average "$destination_dir" "$run_at_server" "$overwrite"
  ;;
  *)
    usage
    exit 1
    ;;
esac