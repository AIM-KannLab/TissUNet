#!/bin/bash
export nnUNet_raw="/media/sda/Elvira/TissUNet"
export nnUNet_results="/media/sda/Elvira/TissUNet/nnUNet_results"
export nnUNet_preprocessed="/media/sda/Elvira/TissUNet"
export CUDA_VISIBLE_DEVICES=0

# Base directories
INPUT_BASE="/media/sda/Elvira/extracranial/data/3d_inputs"
OUTPUT_BASE="/media/sda/Elvira/extracranial/data/3d_outputs"

# List of dataset names (without _reg)
datasets=(
    "abcd_new2023"
    # "abcd_old"
    # "WU1200"
    # "calgary_super"
    # "pixar"
    # "icbm"
    # "healthy_adults_nihm"
    # "pings"
    # "baby"
    # "nyu"
    # "IXI"
    # "sald"
    #"aomic"
)

# Run the prediction script
python predict_multiple.py --datasets "${datasets[@]}" \
                         --input_base "$INPUT_BASE" \
                         --output_base "$OUTPUT_BASE"



# preprocess.py --input /media/sda/Anna/TM2_segmentation/data/t1_mris/pixar_reg \
#                 --output /media/sda/Elvira/extracranial/data/3d_inputs/pixar


# export nnUNet_raw="/media/sda/Elvira/TissUNet"
# export nnUNet_results="/media/sda/Elvira/TissUNet/nnUNet_results"
# export nnUNet_preprocessed="/media/sda/Elvira/TissUNet"
# export CUDA_VISIBLE_DEVICES=0
# nnUNetv2_predict -i /media/sda/Elvira/TissUNet/extracranial/data/3d_inputs/pixar \
#                  -o /media/sda/Elvira/TissUNet/extracranial/data/3d_outputs/pixar \
#                  -d 003 -c 3d_fullres -f all -device cuda


# # nnUNetv2_predict -i mr_pre \
# #                  -o preds \
# #                  -d 003 -c 3d_fullres -f all -device cuda


# preprocess.py --input /media/sda/Anna/TM2_segmentation/data/t1_mris/abcd_new2023_reg \
#                 --output /media/sda/Elvira/extracranial/data/3d_inputs/abcd_new2023