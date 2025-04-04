from __future__ import generators
import os
import shutil
import argparse
import functools
import subprocess
import multiprocessing
from tqdm import tqdm

import itk
import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk

from skimage.transform import resize

from scripts.densenet_regression import DenseNet
from scripts.infer_selection import funcy, get_slice_number_from_prediction
from scripts.preprocess_utils import enhance_noN4

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--input',       '-i',  type=str, required=True, help='Input directory with MR images')
    parser.add_argument('--input_meta',  '-mi', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--meta_output', '-mo', type=str, required=True, help='Output path for meta.csv')
    
    parser.add_argument('--model_weight_path_selection', type=str, default='model_weights/densenet_itmt2.hdf5',
                        help='Path to the model weights for selection. Default: model_weights/densenet_itmt2.hdf5')
    parser.add_argument('--cuda_visible_devices', type=str, default="0",
                        help='CUDA device ID to use. Default: 0')
    parser.add_argument('--dataset', '-d', type=str, required=False, 
                         help='Dataset name to use in output filename (metadata_{dataset}.csv)')                          
    parser.add_argument('--temp_path', '-tp', type=str, default="./temp",
                        help='Path for temporary files. Default: ./temp')
    parser.add_argument('--num_workers', '-n', type=int, default=max(1, os.cpu_count() - 2),
                        help='Number of worker processes to use for multiprocessing. Default: all CPU cores')                         
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise ValueError(f'Input path "{args.input}" does not exist')
    if not os.path.exists(args.input_meta):
        raise ValueError(f'Input metadata path "{args.input_meta}" does not exist')
    return args
            
def configure_devices(cuda_visible_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
    else:   
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
def load_model_selection(model_weight_path_selection):
    model_selection = DenseNet(img_dim=(256, 256, 1), 
                    nb_layers_per_block=16, nb_dense_block=4, growth_rate=16, nb_initial_filters=16, 
                    compression_rate=0.5, sigmoid_output_activation=True, 
                    activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
    model_selection.load_weights(model_weight_path_selection)
    return model_selection

def get_file_name(file_path):
    return os.path.basename(file_path).split(".")[0]

def predict_slice(age = 9, 
                  sex="M",
                  input_path = 'data/t1_mris/nihm_reg/clamp_1193_v1_t1w.nii.gz',
                  path_temp = "data/temp/",
                  model_weight_path_selection = 'model_weights/densenet_itmt2.hdf5'):
    
    model_selection = load_model_selection(model_weight_path_selection)
    print(f'Loaded: {model_weight_path_selection}')  
    
    # Process single image
    patient_id = get_file_name(input_path)
    path_no_z = os.path.join(path_temp, f"{patient_id}_no_z.nii")
    path_z    = os.path.join(path_temp, f"{patient_id}_z.nii")

    # Enhance and z-score normalize image
    image_sitk = sitk.ReadImage(input_path)
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = enhance_noN4(image_array)
    sitk.WriteImage(sitk.GetImageFromArray(image_array), path_no_z)

    # Run external z-score normalization
    subprocess.getoutput(f"zscore-normalize {path_no_z} -o {path_z}")

    # Load normalized image
    image_sitk = sitk.ReadImage(path_z)
    windowed_images = sitk.GetArrayFromImage(image_sitk)

    # Resize image to 256x256
    resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                    preserve_range=True, anti_aliasing=True, mode='constant')
    series = np.dstack([resize_func(im) for im in windowed_images])
    series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])

    # Create Maximum Intensity Projection (MIP) slices
    series_n = [
        np.max(series[slice_idx-2:slice_idx+3, :, :, :], axis=0)
        for slice_idx in range(2, series.shape[0]-2)
    ]
    series_w = np.dstack([funcy(im) for im in series_n])
    series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])

    # Predict the best slice
    predictions = model_selection.predict(series_w)
    slice_label = get_slice_number_from_prediction(predictions)
    return slice_label

def get_file_name(file_path):
    """Extract basename without any extensions from file path"""
    return os.path.splitext(os.path.basename(file_path))[0].split(".")[0]


def process_file(args_tuple):
    """Wrapper function for predict_slice to work with multiprocessing"""
    filename, age, sex, filepath, temp_path, model_weight_path_selection, file_idx, total_files = args_tuple
    
    # Create a unique subfolder in temp_path for each process to avoid conflicts
    process_temp_path = os.path.join(temp_path, f"process_{os.getpid()}")
    os.makedirs(process_temp_path, exist_ok=True)
    
    print(f"[{file_idx+1}/{total_files}] Processing {filename}...")
    try:
        slice_label = predict_slice(
            age=age, 
            sex=sex, 
            input_path=filepath,
            path_temp=process_temp_path,
            model_weight_path_selection=model_weight_path_selection, 
        )
        return filename, slice_label, None
    except Exception as e:
        error_message = f"⚠️ Error processing {filename}: {str(e)}"
        return filename, None, error_message

if __name__ == '__main__':
    args = parse_args()
    configure_devices(args.cuda_visible_devices)
    meta = pd.read_csv(args.input_meta)

    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('.nii.gz')]
    print(f"📄 Found {len(filenames)} files")
    
    # Create temp directory
    temp_path = args.temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
    os.makedirs(temp_path, exist_ok=True)

    # Determine if we're processing the NYU dataset
    is_nyu_dataset = False
    if args.dataset and 'nyu' in args.dataset.lower():
        is_nyu_dataset = True
        print("🔎 Detected NYU dataset - using special filename matching for leading zeros")    
    
    # Add a normalized basename column to metadata for matching
    meta['basename'] = meta['Filename'].astype(str).apply(lambda x: os.path.splitext(os.path.splitext(x)[0])[0] if x.endswith('.nii.gz') 
                                             else (os.path.splitext(x)[0] if x.endswith('.nii') else x))
    
    # Prepare arguments for multiprocessing
    process_args = []
    skipped_files = []
    
    for i, filename in enumerate(filenames):
        # Extract basename without extensions for matching
        basename = os.path.splitext(os.path.splitext(filename)[0])[0]  # Remove .nii.gz
        basename_last_part = os.path.basename(basename)
        
        # Find matching row by basename
        matching_rows = meta[meta['basename'] == basename_last_part]

        # For NYU dataset, try matching without leading zeros if exact match fails
        if is_nyu_dataset and len(matching_rows) == 0:
            basename_no_zeros = basename_last_part.lstrip('0')
            metadata_no_zeros = meta['basename'].str.lstrip('0')
            matching_mask = metadata_no_zeros == basename_no_zeros
            if any(matching_mask):
                matching_rows = meta[matching_mask]
                print(f"✓ Found metadata match for {filename} using zero-stripping")
        
        if len(matching_rows) == 0:
            print(f"⚠️ No metadata match found for {filename}, skipping")
            skipped_files.append(filename)
            continue
            
        age = matching_rows['AGE_M'].values[0]
        sex = matching_rows['SEX'].values[0]
        filepath = os.path.join(args.input, filename)
        
        # Pack all arguments into a tuple for the process_file function
        process_args.append((
            filename, age, sex, filepath, temp_path, 
            args.model_weight_path_selection, i, len(filenames)
        ))
        
    if skipped_files:
        print(f"⚠️ Skipped {len(skipped_files)} files due to missing metadata")
    
    # Process files in parallel
    num_workers = min(args.num_workers, len(process_args))
    print(f"🔄 Processing with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file, process_args),
            total=len(process_args),
            desc="Processing files"
        ))
    
    # Process results
    # for filename, slice_label, error in results:
    #     if error:
    #         print(error)
    #     else:
    #         # Match by basename for updating slice labels
    #         basename = os.path.splitext(os.path.splitext(filename)[0])[0]
    #         meta.loc[meta['basename'] == basename, 'Slice label'] = slice_label
    # Process results
    for filename, slice_label, error in results:
        if error:
            print(error)
        else:
            # Match by basename for updating slice labels
            basename = os.path.splitext(os.path.splitext(filename)[0])[0]  # Remove .nii.gz
            
            # Try direct matching first
            matching_rows = meta[meta['basename'] == basename]
            
            # For NYU dataset, try matching without leading zeros if exact match fails
            if is_nyu_dataset and len(matching_rows) == 0:
                basename_no_zeros = basename.lstrip('0')
                metadata_no_zeros = meta['basename'].str.lstrip('0')
                matching_mask = metadata_no_zeros == basename_no_zeros
                if any(matching_mask):
                    # Update rows where the no-zeros basename matches
                    meta.loc[matching_mask, 'Slice label'] = slice_label
                    print(f"✓ Matched {filename} using zero-stripping")
                    continue
            
            if len(matching_rows) > 0:
                meta.loc[meta['basename'] == basename, 'Slice label'] = slice_label
                print(f"✓ Matched {filename}")
            else:
                print(f"⚠️ No metadata match found for {filename} after processing")
                
                    
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.meta_output), exist_ok=True)

    # Remove rows without a Slice label
    if 'Slice label' in meta.columns:
        original_count = len(meta)
        meta = meta.dropna(subset=['Slice label'])
        removed_count = original_count - len(meta)
        if removed_count > 0:
            print(f"ℹ️ Removed {removed_count} rows without Slice label")
    
    # Create ID column from basename
    meta['ID'] = meta['basename']
    
    # Rename columns
    meta = meta.rename(columns={'AGE_M': 'Age', 'SEX': 'Sex', 'dataset': 'Dataset'})
    
    # Remove unwanted columns
    meta = meta.drop(columns=['SCAN_PATH', 'Filename', 'basename'], errors='ignore')
    # Convert to integer
    meta['Slice label'] = meta['Slice label'].astype(int)

    # Extract dataset name without suffix ("_reg")
    dataset_name = args.dataset.split('_')[0] if args.dataset else "unknown"
        
    meta.to_csv(os.path.join(args.meta_output, f'metadata_{dataset_name}.csv'), index=False)
    print(f'✅ metadata_{dataset_name}.csv saved with slice labels')   