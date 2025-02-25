import os
import sys
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json

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

def extract_img_info_json(img: nib.Nifti1Image):
    info = {
        "affine": img.affine.tolist(),
        "data_shape": list(img.shape),
        "data_dtype": str(img.get_data_dtype()),
        "spatial_resolution": [float(x) for x in img.header.get_zooms()],
    }
    return info

def main(args):
    nii_files = [f for f in os.listdir(args.preds_input) if f.endswith('.nii.gz')]
    total_samples = len(nii_files)
    print(f"🔍 Found {total_samples} NIfTI files in directory '{args.preds_input}'.")

    metrics = {}
    for sample_name in tqdm(sorted(nii_files), desc="🖼️ Processing NIfTI files"):
        nii_path = os.path.join(args.preds_input, sample_name)
        img = nib.load(nii_path)
        img_data = img.get_fdata().astype(np.uint8)
        if not np.all(np.equal(np.mod(img_data, 1), 0)): 
            print("Warning: Non-integer values found in NIfTI file")
        labels = np.unique(img_data)
        
        volumes = {}
        for label in labels:
            mask = (img_data == label).astype(np.uint8)
            volume = int(np.sum(mask))
            volumes[str(int(label))] = volume
        
        metrics[sample_name] = {
            "path": nii_path,
            "sample_name": sample_name,
            "volumes": volumes,
            "img_info": extract_img_info_json(img),
        }

    with open(args.metrics_output, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✅ Processing complete! Processed {total_samples} samples in total.")
    print(f"📁 Results saved to '{args.metrics_output}'.")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
