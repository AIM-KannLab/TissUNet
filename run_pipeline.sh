#!/bin/bash

set -e  # Exit on error

# ---------------- Parse Arguments ----------------
if [ "$#" -lt 3 ]; then
    echo "❌ Usage: bash run_pipeline.sh <in_dir> <out_dir> <cpu/cuda> [--no-register] [--cleanup]"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
DEVICE=$3
NO_REGISTER=false
CLEANUP=false

# Optional flags
for arg in "$@"; do
    if [ "$arg" == "--no-register" ]; then
        NO_REGISTER=true
    elif [ "$arg" == "--cleanup" ]; then
        CLEANUP=true
    fi
done 

echo "📂 Input MR folder: $INPUT_DIR"
echo "📁 Output base folder: $OUTPUT_DIR"
echo "🖥️ Device: $DEVICE"
echo "🚫 Registration: $NO_REGISTER"
echo "🧹 Cleanup: $CLEANUP"

# ---------------- Create Directory Structure ----------------
# echo "📁 Creating output folders..."
# mkdir -p "$OUTPUT_DIR"/{mr_pre,mr_post,mr_post_def,preds,preds_post,preds_post_def,model_weights,nnUNet_results,nnUNet_raw,nnUNet_preprocessed}

# Copy meta.csv if present
# if [ -f "$INPUT_DIR/meta.csv" ]; then
#     cp "$INPUT_DIR/meta.csv" "$OUTPUT_DIR/mr_pre/meta.csv"
#     echo "📋 meta.csv copied to $OUTPUT_DIR/mr_pre/"
# else
#     echo "⚠️ meta.csv not found in $INPUT_DIR. Skipping slice prediction."
# fi

# ---------------- Download Weights ----------------
# echo "⬇️ Downloading model weights..."
# python download_weights.py

# ---------------- Preprocess ----------------
echo "====================================="
echo "=============STEP 1=================="
echo "====================================="
echo "🧼 Preprocessing input data..."
PREPROCESS_CMD="python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR/mr_pre"
if [ "$NO_REGISTER" = true ]; then
    PREPROCESS_CMD+=" --no-register"
fi
# Print the command for debugging
eval $PREPROCESS_CMD

# ---------------- Predict ----------------
echo "====================================="
echo "=============STEP 2.1================"
echo "====================================="
echo "🔮 Running predictions..."
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

PREDICT_CMD="python predict.py -i $OUTPUT_DIR/mr_pre -o $OUTPUT_DIR/preds -d $DEVICE"
if [ "$CLEANUP" = true ]; then
    PREDICT_CMD+=" --cleanup"
fi
eval $PREDICT_CMD

# ---------------- Predict Slices ----------------
echo "====================================="
echo "=============STEP 2.2================"
echo "====================================="
if [ -f "$OUTPUT_DIR/mr_pre/meta.csv" ]; then
    echo "🧠 Running slice prediction..."
    python predict_slices.py -i "$OUTPUT_DIR/mr_pre" \
                             -mi "$OUTPUT_DIR/mr_pre/meta.csv" \
                             -mo "$OUTPUT_DIR/preds/meta.csv"
fi

# ---------------- Postprocess ----------------
echo "====================================="
echo "=============STEP 3=================="
echo "====================================="
echo "🧽 Postprocessing predictions..."
python postprocess.py -mi "$OUTPUT_DIR/mr_pre" \
                      -pi "$OUTPUT_DIR/preds" \
                      -mo "$OUTPUT_DIR/mr_post" \
                      -po "$OUTPUT_DIR/preds_post"

python postprocess.py -mi "$OUTPUT_DIR/mr_pre" \
                      -pi "$OUTPUT_DIR/preds" \
                      -mo "$OUTPUT_DIR/mr_post_def" \
                      -po "$OUTPUT_DIR/preds_post_def" \
                      --deface

# ---------------- Compute Metrics ----------------
echo "====================================="
echo "=============STEP 4=================="
echo "====================================="
echo "📊 Computing metrics..."
python compute_metrics.py -pi "$OUTPUT_DIR/preds_post" \
                          -mo "$OUTPUT_DIR/preds_post/metrics.csv"

python compute_metrics.py -pi "$OUTPUT_DIR/preds_post_def" \
                          -mo "$OUTPUT_DIR/preds_post_def/metrics.csv"

echo "✅ Pipeline complete! All results saved in '$OUTPUT_DIR/'"
