#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Base directories
INPUT_BASE="/media/sda/Anna/TM2_segmentation/data/t1_mris"
OUTPUT_BASE="/media/sda/Elvira/extracranial/data/supp_data"
TEMP_BASE="/media/sda/Elvira/extracranial/data/temp"

# Define dataset-to-metadata mapping with a bash associative array
declare -A metadata_paths
metadata_paths=(
    ["28_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_28.csv"
    ["long579_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_long579.csv"
    ["PediatricMRI_DEMOGRAPHICS_reg_clamp"]="/media/sda/Anna/TM2_segmentation/data/Dataset_PedMRI_clamp.csv"
    ["dexa_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_tmt2_dexa_v2.csv"
    ["aomic_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_aomic.csv"
    ["abcd_new2023_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_abcd_new2023.csv"
    ["abcd_old_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_abcd_old.csv"
    ["WU1200_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_WU1200.csv"
    ["calgary_super_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_calgary.csv"
    ["pixar_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_pixar_v2.csv"
    ["icbm_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_icbm.csv"
    ["healthy_adults_nihm_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_healthy_adults_nihm.csv"
    ["pings_registered"]="/media/sda/Anna/TM2_segmentation/data/Dataset_ping.csv"
    ["baby_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_BABY.csv"
    ["nyu_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_nyu.csv"
    ["IXI_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_IXI.csv"
    ["sald_reg"]="/media/sda/Anna/TM2_segmentation/data/Dataset_sald.csv"
    # add for abide
    ["abide_registered"]="/media/sda/Anna/TM2_segmentation/data/Dataset_abide.csv"
)

datasets=(
    # "pixar_reg" # done    
    # "28_reg"
    # "long579_reg"
    # "abide_registered"
    #"PediatricMRI_DEMOGRAPHICS_reg_clamp" #done
    #"dexa_reg" #done
    #"aomic_reg" #done
    # "abcd_new2023_reg"
    # "abcd_old_reg" #done
    #"WU1200_reg" #done
    # "calgary_super_reg" # done
    "icbm_reg" # done
    #"healthy_adults_nihm_reg" # done
    #"pings_registered" # done
    #"baby_reg" # done
    # "nyu_reg" # done
    #"IXI_reg" # done 
    #"sald_reg" # done
)

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    # Get the correct metadata path from the associative array
    metadata_path=${metadata_paths["$dataset"]}
    
    # Check if metadata path exists for this dataset
    if [ -z "$metadata_path" ]; then
        echo "Warning: No metadata path found for dataset '$dataset'. Skipping."
        continue
    fi
    
    python predict_slices.py --input "$INPUT_BASE/$dataset" \
                            --meta_output "$OUTPUT_BASE" \
                            --input_meta "$metadata_path" \
                            --dataset "$dataset" \
                            --cuda_visible_devices 0 \
                            --temp_path "$TEMP_BASE/$dataset"
done

echo "All datasets processed!"