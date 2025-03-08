#!/bin/bash

# Base directory where all your datasets are stored
BASE_DIR="/media/sda/Anna/TM2_segmentation/data/t1_mris"

# Check output directory permissions
OUTPUT_DIR="/media/sda/Elvira/extracranial/data/3d_inputs"

# First check if we can create/access the directory
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Check if directory is writable
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "Error: No write permission in $OUTPUT_DIR"
    echo "Current permissions:"
    ls -ld "$OUTPUT_DIR"
    exit 1
fi

# List of dataset directories
datasets=(
    #"aomic_reg" #done
    #"abcd_new2023_reg"
    # "abcd_old_reg" #done, new to do
    # "WU1200_reg" #done
    "calgary_super_reg" # done
    #"pixar_reg" # done
    # "icbm_reg" # done
    # "healthy_adults_nihm_reg" # done
    "pings_registered" # done
    # "baby_reg" # done
    # "nyu_reg" # done
    # "IXI_reg" # done 
    # "sald_reg" # done
)

# Construct the full paths and run the preprocessing
input_dirs=""
for dataset in "${datasets[@]}"; do
    input_dirs="$input_dirs $BASE_DIR/$dataset"
done
python preprocess_multiple.py --input_dirs $input_dirs