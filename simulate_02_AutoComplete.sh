#!/bin/bash
source activate ML

# Parse input parameters
is_server="$1"
Data_to_analyze="$2"
AutoCompleteFit="$3/fit.py"
Results="$4"

lr="$5"
batch_size="$6"
epochs="$7"
momentum="$8"
encoding_ratio="$9"
copymask_amount="${10}"
simulate_missing="${11}"
depth="${12}"
overwrite="${13}" 


# Ensure the Results directory exists; if not, create it
if [ ! -d "$Results" ]; then
    echo "Results directory does not exist. Creating it at: $Results"
    mkdir -p "$Results"
fi

# Strip the '.csv' extension from Data_to_analyze and generate filenames with suffixes
base_name=$(basename "$Data_to_analyze" .csv)

# Define a list of seeds for the loop
seeds=(1 2 3 4 5 6 7 8 9 10)

# Loop through each seed and create a corresponding subdirectory
for seed in "${seeds[@]}"; do
    # Create a subdirectory for the current seed
    seed_dir="${Results}/seed_${seed}"
    mkdir -p "$seed_dir"
    
    # Generate filenames within the seed subdirectory
    model_filename="${seed_dir}/${base_name}_model.pth"
    impute_filename="${seed_dir}/${base_name}_imputed"

    # Check for existing file and overwrite setting
    if [ "$overwrite" == "0" ] && [ -f "$model_filename" ]; then
        echo "File ${model_filename} already exists and overwrite is set to 0. Skipping submission for seed $seed."
        continue
    fi

    # Print the filenames for debugging
    echo "Model will be saved as: ${model_filename}"
    echo "Imputed data will be saved as: ${impute_filename}"

    # Check if the first argument is "1", indicating submission to a cluster with sbatch
    if [ "$1" == "1" ]; then
        echo "Submitting the job to the cluster using sbatch for seed $seed."

        # Create a Slurm batch script for submission
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=AutoCompleteFit_${seed}
#SBATCH --output=${seed_dir}/slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem=16G
#SBATCH --time=1:00:00

python "$AutoCompleteFit" \
    "$Data_to_analyze" \
    --id_name ID \
    --lr "$lr" \
    --batch_size "$batch_size" \
    --val_split 0.8 \
    --device cpu:0 \
    --epochs "$epochs" \
    --momentum "$momentum" \
    --output "${impute_filename}" \
    --encoding_ratio "$encoding_ratio" \
    --depth "$depth" \
    --copymask_amount "$copymask_amount" \
    --num_torch_threads 8 \
    --simulate_missing "$simulate_missing" \
    --bootstrap \
    --quality \
    --save_imputed \
    --seed "$seed" \
    --save_model_path "${model_filename}"
EOF

    else
        # Run the Python script directly
        echo "Running the Python script directly for seed $seed."

        python "$AutoCompleteFit" \
            "$Data_to_analyze" \
            --id_name ID \
            --lr "$lr" \
            --batch_size "$batch_size" \
            --val_split 0.8 \
            --device cpu:0 \
            --epochs "$epochs" \
            --momentum "$momentum" \
            --output "${impute_filename}" \
            --encoding_ratio "$encoding_ratio" \
            --depth "$depth" \
            --copymask_amount "$copymask_amount" \
            --num_torch_threads 8 \
            --simulate_missing "$simulate_missing" \
            --bootstrap \
            --quality \
            --save_imputed \
            --seed "$seed" \
            --save_model_path "${model_filename}"
    fi
done
