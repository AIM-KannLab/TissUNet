import os
import sys
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Postprocessing')
    parser.add_argument('--preds_input', '-pi', type=str, required=True, help='Predictions input directory')
    parser.add_argument('--metrics_output', '-mo', type=str, required=True, help='Metrics output file')
    args = parser.parse_args()
    if not os.path.exists(args.preds_input) or not os.path.isdir(args.preds_input):
        print(f"Directory not found: {args.preds_input}")
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
    label_map = {
        0: "background",
        1: "brain",
        2: "temporalis",
        3: "other_muscles",
        4: "skull",
        5: "subcutaneous_fat"
    }

    nii_files = [f for f in os.listdir(args.preds_input) if f.endswith('.nii.gz')]
    total_samples = len(nii_files)
    print(f"🔍 Found {total_samples} NIfTI files in directory '{args.preds_input}'.")

    metrics = {}
    for file_name in tqdm(sorted(nii_files), desc="🖼️ Processing NIfTI files"):
        nii_path = os.path.join(args.preds_input, file_name)
        img = nib.load(nii_path)
        img_data = img.get_fdata().astype(np.uint8)
        if not np.all(np.equal(np.mod(img_data, 1), 0)): 
            print(f"⚠️ Warning: Non-integer values found in {file_name}")
        labels = np.unique(img_data)
        
        volumes = {}
        for label in labels:
            if int(label) == 0:
                continue  # Skip background
            mask = (img_data == label).astype(np.uint8)
            volume = int(np.sum(mask))
            volumes[str(int(label))] = volume
        
        sample_name = file_name.replace('.nii.gz', '')
        metrics[sample_name] = {
            # "path": nii_path,
            # "sample_name": sample_name,
            "volumes": volumes,
        }
    
    df_meta = pd.read_csv(os.path.join(args.preds_input, 'meta.csv'))
    records = []
    for key, value in metrics.items():
        meta_data = df_meta[df_meta['sample_name'] == key]
        flat_record = {}
        if not meta_data.empty:
            flat_record['file_name'] = meta_data.iloc[0]['file_name']
            flat_record['sample_name'] = meta_data.iloc[0]['sample_name']
            flat_record['age'] = meta_data.iloc[0]['age']
            flat_record['sex'] = meta_data.iloc[0]['sex']
            flat_record['slice_idx'] = meta_data.iloc[0]['slice_idx']
        
        flat_record.update({k: v for k, v in value.items() if k != 'volumes' and k != 'img_info'})

        for vol_key, vol_value in value.get('volumes', {}).items():
            label_int = int(vol_key)
            if label_int == 0:
                continue  # Also skip background here if somehow included
            tissue_name = label_map.get(label_int, f"label_{label_int}")
            flat_record[f'vol_{tissue_name}'] = vol_value

        for info_key, info_value in value.get('img_info', {}).items():
            flat_record[f'img_info_{info_key}'] = info_value

        records.append(flat_record)

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(args.metrics_output, index=False)

    print(f"✅ Processing complete! Processed {total_samples} samples in total.")
    print(f"📁 Results saved to '{args.metrics_output}'.")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
