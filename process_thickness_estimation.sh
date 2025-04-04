#!/bin/bash

# Define the script path
#SCRIPT_PATH="SkullThickness/scripts/main_thickness_estimation.py"

# Define the datasets array (using the same list as in predict_slices_multiple.sh)
datasets=(
    # "abcd_new2023"
    # # "abcd_old"
    # "WU1200"
    # "calgary" # done
    "pixar" # done
    # "icbm"
    # "healthy_adults_nihm"
    # "PediatricMRI"
    # "pings"
    # "baby"
    # "nyu"
    # "IXI"
    # "sald"
    # "aomic"
    # "long579"
    # "abide"
)

# Process each dataset in the array
for dataset in "${datasets[@]}"; do
    # Skip commented out datasets
    if [[ $dataset == \#* ]]; then
        echo "Skipping commented dataset: ${dataset#\#}"
        continue
    fi
    
    echo "==============================================="
    echo "Processing dataset: $dataset"
    echo "Start time: $(date)"
    echo "==============================================="
    
    # Run the Python script with the current dataset
    # specify the paths to the image input, metadata, csv output and plot output
    python scripts/main_thickness_estimation.py --dataset "$dataset"
    
    # Check if the script executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed dataset: $dataset"
    else
        echo "Error processing dataset: $dataset"
    fi
    
    echo "End time: $(date)"
    echo ""
done

echo "All datasets have been processed!"