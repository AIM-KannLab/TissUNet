from __future__ import generators
import os
import shutil
import argparse
import functools
import subprocess

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

if __name__ == '__main__':
    args = parse_args()
    configure_devices(args.cuda_visible_devices)
    meta = pd.read_csv(args.input_meta)

    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('.nii.gz')]
    # Print "Found n files" where n is the number of files with emoji
    print(f"📄 Found {len(filenames)} files")
    print()
    temp_path = os.path.join(args.input, 'temp')
    shutil.rmtree(temp_path, ignore_errors=True)
    os.makedirs(temp_path, exist_ok=True)
    for i, filename in enumerate(filenames):
        # try:
        print(f"[{i+1}/{len(filenames)}] Processing {filename}...")
        row = meta[meta['filename'] == filename]
        print(row)
        age = row['age'].values[0]
        sex = row['sex'].values[0]
        filepath = os.path.join(args.input, filename)
        
        slice_label = predict_slice(
            age=age, 
            sex=sex, 
            input_path=filepath,
            path_temp=temp_path,
            model_weight_path_selection=args.model_weight_path_selection, 
        )
        meta.loc[meta['filename'] == filename, 'slice_label'] = slice_label
        print()
        # except Exception as e:
        #     print(f"⚠️ Error processing {filename}: {str(e)}")
        #     print(f"Skipping this file and continuing with the next one.")
        #     print()
        #     continue

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.meta_output), exist_ok=True)
    
    # Remove rows without a slice_label
    if 'slice_label' in meta.columns:
        original_count = len(meta)
        meta = meta.dropna(subset=['slice_label'])
        removed_count = original_count - len(meta)
        if removed_count > 0:
            print(f"ℹ️ Removed {removed_count} rows without slice_label")
    
    meta.to_csv(args.meta_output, index=False)
    print(f'✅ {args.meta_output} saved with slice labels')