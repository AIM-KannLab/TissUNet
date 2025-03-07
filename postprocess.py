# Postprocessing
import os
import argparse
import json
import numpy as np
import scipy.ndimage
import nibabel as nib

def parse_args():
    parser = argparse.ArgumentParser(description='Postprocessing')
    parser.add_argument('--mr_input',  '-mi', type=str, required=True,  help='MRI input directory')
    parser.add_argument('--preds_input',  '-pi', type=str, required=True,  help='Predicitons input directory')
    parser.add_argument('--mr_output', '-mo', type=str, required=True, help='MRI output directory')
    parser.add_argument('--preds_output', '-po', type=str, required=True, help='Predicitons output directory')
    parser.add_argument('--deface', '-d', action='store_true', help='Deface the MRI and predictions')
    args = parser.parse_args()
    return args

def keep_largest_component(seg: np.ndarray) -> np.ndarray:
    assert np.array_equal(np.unique(seg), [0, 1]), 'Segmentation mask must be binary (0 and 1)'

    labeled_array, num_features = scipy.ndimage.label(seg)  # Label connected components
    if num_features == 0:
        return seg  # Return the same mask if no components exist

    component_sizes = scipy.ndimage.sum(seg, labeled_array, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1

    cleaned_seg = (labeled_array == largest_label).astype(np.uint8)
    return cleaned_seg

def deface_nii_left(mri: np.array, brain_seg: np.array, background_value: int):
    brain_seg_mask = brain_seg > 0
    first_col_idxs_with_nonzero_value = []
    for z in range(mri.shape[2]):
        slice_mask = brain_seg_mask[:, :, z]
        first_column_with_nonzero_value = np.argmax(np.sum(np.abs(slice_mask), axis=0) > 0)
        first_col_idxs_with_nonzero_value.append(first_column_with_nonzero_value)
    
    lobe_slice_idx = np.argmin([i if i > 0 else np.inf for i in first_col_idxs_with_nonzero_value])
    for z in range(0, lobe_slice_idx):
        first_col_idxs_with_nonzero_value[z] = first_col_idxs_with_nonzero_value[lobe_slice_idx]
    
    mri_defaced = mri.copy()
    for z, col_idx in enumerate(first_col_idxs_with_nonzero_value):
        if col_idx == 0 and z > lobe_slice_idx:
            col_idx = mri.shape[1]
        mri_defaced[:, :col_idx, z] = background_value
    return mri_defaced

def deface_mri_left(mri: np.array, brain_seg: np.array):
    return deface_nii_left(mri, brain_seg, background_value=0)

def main(args):
    assert os.path.exists(args.mr_input), f'MRI input directory not found: {args.mr_input}'
    assert os.path.exists(args.preds_input), f'Predictions input directory not found: {args.preds_input}'
    ds_json_path = os.path.join(args.preds_input, 'dataset.json')
    assert os.path.exists(ds_json_path), f'dataset.json file not found in: {args.preds_input}'
    ds_json = json.load(open(ds_json_path))
    brain_val = ds_json['labels']['brain']
    print(f'Creating output directory {args.mr_output} ...')
    os.makedirs(args.mr_output, exist_ok=True)
    print(f'Creating output directory {args.preds_output} ...')
    os.makedirs(args.preds_output, exist_ok=True)
    print("Scanning input directories ...")
    paths_pairs = []
    pred_filenames = [fn for fn in os.listdir(args.preds_input) if fn.endswith('.nii.gz')]
    for file in sorted(pred_filenames):
        if not file.endswith('.nii.gz'):
            print(f"Skipping {file} (not a nii.gz file)")
            continue
        sample_name = file.split('.')[0]
        mr_path = os.path.join(args.mr_input, f'{sample_name}_0000.nii.gz')
        pred_path = os.path.join(args.preds_input, file)
        assert os.path.exists(mr_path), f'MR file not found: {mr_path}'
        assert os.path.exists(pred_path), f'Predictions file not found: {pred_path}'
        paths_pairs.append((mr_path, pred_path))
    print(f"Found {len(paths_pairs)} pairs of files. Processing ...")
    for i, (mr_path, pred_path) in enumerate(paths_pairs):
        print(f"[{i+1}/{len(paths_pairs)}] Processing {mr_path} and {pred_path} ...")
        mr = nib.load(mr_path)
        mr_data = mr.get_fdata()
        pred = nib.load(pred_path)
        pred_data = pred.get_fdata()
        
        # Copy data
        mr_out_data   = mr_data.copy()
        pred_out_data = pred_data.copy()
        # Filter brain mask
        brain_mask = (pred_data == brain_val).astype(np.uint8)
        brain_mask_filtered = keep_largest_component(brain_mask)
        pred_out_data[(pred_data == brain_val) & (brain_mask_filtered == 0)] = 0
        
        if args.deface:
            # Deface MRI
            mr_out_data = deface_mri_left(mr_out_data, brain_mask_filtered)
            pred_out_data = deface_nii_left(pred_out_data, brain_mask_filtered, background_value=0)
        
        # Create output paths
        mr_out_path   = os.path.join(args.mr_output, os.path.basename(mr_path))
        pred_out_path = os.path.join(args.preds_output, os.path.basename(pred_path))
        # Create output files
        mr_out_file   = nib.Nifti1Image(mr_out_data, mr.affine) 
        pred_out_file = nib.Nifti1Image(pred_out_data, pred.affine)
        # Save output files
        nib.save(mr_out_file, mr_out_path)
        nib.save(pred_out_file, pred_out_path)
        
    print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)