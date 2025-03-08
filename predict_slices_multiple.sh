#!/bin/bash

# Set script to exit on error
set -e

# Directory paths
INPUT_DIR="/media/sda/Elvira/extracranial/data/3d_outputs/pixar"
OUTPUT_DIR="/media/sda/Elvira/extracranial/data/supp_data/pixar"
METADATA_PATH="/media/sda/Anna/TM2_segmentation/data/Dataset_pixar_v2.csv"
MODEL_WEIGHTS="model_weights/densenet_itmt2.hdf5"  # Default path, change if needed

# Set CUDA device
#export CUDA_VISIBLE_DEVICES=0
# Add CUDA environment variables
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}


# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if files and directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file does not exist: $METADATA_PATH"
    exit 1
fi

echo "🚀 Starting slice prediction..."
echo "📁 Input directory: $INPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📄 Metadata file: $METADATA_PATH"
echo "🖥️ Using CUDA device: $CUDA_VISIBLE_DEVICES"

# Run the prediction script
python predict_slices.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --metadata "$METADATA_PATH" \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --model_weight_path_selection "$MODEL_WEIGHTS"

echo "✅ Prediction completed!"