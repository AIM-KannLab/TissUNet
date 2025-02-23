import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import imea
import warnings
import logging

warning_log_count = 0
logging.basicConfig(
    filename='imea_warnings.txt',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    global warning_log_count
    log_message = f"{category.__name__} in {filename} at line {lineno}: {message}"
    logging.warning(log_message)
    warning_log_count += 1

warnings.showwarning = custom_showwarning

def parse_args():
    parser = argparse.ArgumentParser(description='Postprocessing')
    parser.add_argument('--preds_input', '-pi', type=str, required=True, help='Predictions input directory')
    parser.add_argument('--metrics_output', '-mo', type=str, required=True, help='Metrics output file')
    args = parser.parse_args()
    if not os.path.exists(args.preds_input) or not os.path.isdir(args.preds_input):
        print(f"Directory not found: {args.preds_input}")
        sys.exit(1)
    if os.path.exists(args.metrics_output):
        print(f"File already exists: {args.metrics_output}")
        sys.exit(1)
    return args

def main(args):
    global warning_log_count
    args.preds_input = "preds_post"
    nii_files = [f for f in os.listdir(args.preds_input) if f.endswith('.nii.gz')]
    total_samples = len(nii_files)
    print(f"🔍 Found {total_samples} NIfTI files in directory '{args.preds_input}'.")
    
    df = pd.DataFrame()
    for sample_idx, sample_name in enumerate(nii_files, start=1):
        nii_path = os.path.join(args.preds_input, sample_name)
        img = nib.load(nii_path)
        img_data = img.get_fdata().astype(np.uint8)
        assert np.all(np.equal(np.mod(img_data, 1), 0)), "Non-integer values found in NIfTI file"
        header = img.header
        spatial_resolution_x, spatial_resolution_y, spatial_resolution_z = header.get_zooms()
        assert spatial_resolution_x == spatial_resolution_y, "Spatial resolution in X and Y dimensions must be equal"
        labels = [l for l in np.unique(img_data) if l != 0]
        
        for label in labels:
            tqdm_bar = tqdm(range(0, img_data.shape[2], 50),
                            desc=f"[{sample_idx}/{total_samples}, {label}] {sample_name}",
                            leave=True)
            for idx in tqdm_bar:
                img_slice = img_data[:, :, idx]
                mask = (img_slice == label).astype(np.uint8)
                if np.sum(mask) == 0:
                    continue

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    df_2d, df_3d = imea.shape_measurements_3d(mask, 0.5, spatial_resolution_x, spatial_resolution_z)
                    for w in caught_warnings:
                        warning_msg = (
                            f"Sample: {sample_name}, Label: {label}, Slice: {idx}, "
                            f"File Path: {nii_path} -- {w.category.__name__}: {w.message}"
                        )
                        logging.warning(warning_msg)
                        warning_log_count += 1
                
                volume_manual = np.sum(mask)
                df_ = pd.concat([df_2d, df_3d], axis=1)
                df_.insert(0, 'volume_manual', volume_manual)
                df_.insert(0, 'spatial_resolution_z', spatial_resolution_z)
                df_.insert(0, 'spatial_resolution_y', spatial_resolution_y)
                df_.insert(0, 'spatial_resolution_x', spatial_resolution_x)
                df_.insert(0, 'object_id', df_.index)
                df_.insert(0, 'slice_idx', idx)
                df_.insert(0, 'label', label)
                df_.insert(0, 'sample_name', sample_name)
                
                df = pd.concat([df, df_], axis=0)

    # Save results to CSV with an added column for the file path.
    df.insert(0, 'path', df['sample_name'].apply(lambda name: os.path.join(args.preds_input, name)))
    df.to_csv(args.metrics_output, index=False)
    print(f"\n✅ Processing complete! Processed {total_samples} samples in total.")
    print(f"📁 Results saved to '{args.metrics_output}'.")
    if warning_log_count > 0:
        print(f"⚠️ {warning_log_count} warnings logged to 'imea_warnings.txt'.")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
