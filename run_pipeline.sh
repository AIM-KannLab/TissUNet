#!/bin/bash

set -e  # Exit on error

# ---------------- Parse Arguments ----------------
if [ "$#" -lt 3 ]; then
    echo "❌ Usage: bash run_pipeline.sh <in_dir> <out_dir> <cpu/cuda> [--meta <meta_path>] [--no-register] [--no-predict-slices] [--cleanup] [--thickness]"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
DEVICE=$3
NO_REGISTER=false
NO_PREDICT_SLICES=false
CLEANUP=false
RUN_THICKNESS=false
META_PATH=""

# Optional flags
shift 3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-register)
            NO_REGISTER=true
            shift
            ;;
        --no-predict-slices)
            NO_PREDICT_SLICES=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --thickness)
            RUN_THICKNESS=true
            shift
            ;;
        --meta)
            META_PATH="$2"
            shift 2
            ;;
        *)
            echo "⚠️ Unknown option: $1"
            shift
            ;;
    esac
done 

echo "📂 Input MR folder: $INPUT_DIR"
echo "📁 Output base folder: $OUTPUT_DIR"
echo "🖥️ Device: $DEVICE"
if [ "$NO_REGISTER" = true ]; then echo "👨‍🦲 Registration: false"; else echo "👨‍🦲 Registration: true"; fi
if [ "$NO_PREDICT_SLICES" = true ]; then echo "🍕 Slice prediction: false"; else echo "🍕 Slice prediction: true"; fi
echo "🧹 Cleanup: $CLEANUP"
echo "📏 Thickness estimation: $RUN_THICKNESS"
if [ -n "$META_PATH" ]; then echo "🧾 Using external meta path: $META_PATH"; fi
# if $META_PATH is not set, use the $INPUT_DIR/meta.csv
if [ -z "$META_PATH" ]; then
    META_PATH="$INPUT_DIR/meta.csv"
    echo "🗂️ Meta path: $META_PATH"
fi

# ---------------- Checks ----------------
# If $NO_PREDICT_SLICES is true, check if the meta.csv file contains the 'slice_idx' column
if [ "$NO_PREDICT_SLICES" = true ]; then
    if ! grep -q "slice_idx" "$META_PATH"; then
        echo "❌ Error: --no-predict-slices is set but $META_PATH does not contain the 'slice_idx' column."
        echo "Please remove the --no-predict-slices flag or add the 'slice_idx' column to the meta.csv file."
        exit 1
    fi
fi

# ---------------- Preprocess ----------------
echo "====================================="
echo "=============STEP 1=================="
echo "====================================="
echo "🧼 Preprocessing input data..."
PREPROCESS_CMD="python preprocess.py -i $INPUT_DIR -o $OUTPUT_DIR/mr_pre --meta $META_PATH"
if [ "$NO_REGISTER" = true ]; then
    PREPROCESS_CMD+=" --no-register"
fi
eval $PREPROCESS_CMD

# ---------------- Predict ----------------
echo "====================================="
echo "=============STEP 2.1================"
echo "====================================="
echo "🔮🧠 Running predictions..."
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"

PREDICT_CMD="python predict.py -i $OUTPUT_DIR/mr_pre -o $OUTPUT_DIR/preds -d $DEVICE"
if [ "$CLEANUP" = true ]; then
    PREDICT_CMD+=" --cleanup"
fi
echo "Running command: $PREDICT_CMD"
eval $PREDICT_CMD

# ---------------- Predict Slices ----------------
if [ "$NO_PREDICT_SLICES" = true ]; then
    echo "🍕 Skipping slice prediction..."
else
    echo "====================================="
    echo "=============STEP 2.2================"
    echo "====================================="
    echo "🔮🍕 Running slice prediction..."
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
                          -mo "$OUTPUT_DIR/preds_post/meta.csv"

python compute_metrics.py -pi "$OUTPUT_DIR/preds_post_def" \
                          -mo "$OUTPUT_DIR/preds_post_def/meta.csv"

# ---------------- Thickness Estimation ----------------
if [ "$RUN_THICKNESS" = true ]; then
    echo "====================================="
    echo "=============STEP 5=================="
    echo "====================================="
    echo "📏 Running thickness estimation..."
    python scripts/main_thickness_estimation.py --dataset "demo" \
                                                --lookup-slice-table "$OUTPUT_DIR/preds/meta.csv" \
                                                --csv-output-dir "$OUTPUT_DIR/results_thickness" \
                                                --plot-output-dir "$OUTPUT_DIR/results_thickness" \
                                                --processed-image-dir "$OUTPUT_DIR/preds"
fi

# ---------------------- Finish ----------------
echo "====================================="
echo "=============DONE====================="
echo "====================================="
echo "✅ Pipeline complete! All results saved in '$OUTPUT_DIR/'"
if [ "$RUN_THICKNESS" = true ]; then
    echo "📏 Thickness estimation results saved in '$OUTPUT_DIR/results_thickness/'"
fi