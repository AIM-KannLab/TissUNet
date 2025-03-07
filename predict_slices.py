from __future__ import generators
import os
import argparse
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import resize

from scripts.densenet_regression import DenseNet
from scripts.infer_selection import funcy, get_slice_number_from_prediction
from scripts.preprocess_utils import load_nii

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--input',  '-i', type=str, required=True,  help='Input directory with MR images and meta.csv')
    parser.add_argument('--output', '-o', type=str, required=False, help='Output path for meta.csv')
    parser.add_argument('--model_weight_path_selection', type=str, default='model_weights/densenet_itmt2.hdf5',
                        help='Path to the model weights for selection')
    parser.add_argument('--cuda_visible_devices', type=str, default="0",
                        help='CUDA device ID to use')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise ValueError('Input path does not exist')
    if not os.path.exists(os.path.join(args.input, 'meta.csv')):
        raise ValueError('meta.csv does not exist in input path')
    if not args.output:
        args.output = args.input
    return args

def predict_slice(age=9, 
                  sex="M", 
                  input_path='data/t1_mris/nihm_reg/clamp_1193_v1_t1w.nii.gz',
                  model_weight_path_selection='model_weights/densenet_itmt2.hdf5', 
                  cuda_visible_devices="0"):
    
    print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
    else:   
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    # Load DenseNet model for slice selection
    model_selection = DenseNet(img_dim=(256, 256, 1), 
                               nb_layers_per_block=16, nb_dense_block=4, growth_rate=16, 
                               nb_initial_filters=16, compression_rate=0.5, 
                               sigmoid_output_activation=True, activation_type='relu', 
                               initializer='glorot_uniform', output_dimension=1, batch_norm=True)
    model_selection.load_weights(model_weight_path_selection)
    print('Loaded:', model_weight_path_selection)  
    
    # Load image
    image, affine = load_nii(input_path)
    
    # Resize image to match model input size (256x256)
    resize_func = functools.partial(resize, output_shape=model_selection.input_shape[1:3],
                                    preserve_range=True, anti_aliasing=True, mode='constant')
    series = np.dstack([resize_func(im) for im in image])
    series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
    
    # Generate multi-intensity projection (MIP) of 5 slices
    series_n = []
    for slice_idx in range(2, np.shape(series)[0] - 2):
        im_array = np.zeros((256, 256, 1, 5))
        im_array[:,:,:,0] = series[slice_idx-2,:,:,:].astype(np.float32)
        im_array[:,:,:,1] = series[slice_idx-1,:,:,:].astype(np.float32)
        im_array[:,:,:,2] = series[slice_idx,:,:,:].astype(np.float32)
        im_array[:,:,:,3] = series[slice_idx+1,:,:,:].astype(np.float32)
        im_array[:,:,:,4] = series[slice_idx+2,:,:,:].astype(np.float32)
        im_array = np.max(im_array, axis=3)
        series_n.append(im_array)
        
    series_w = np.dstack([funcy(im) for im in series_n])
    series_w = np.transpose(series_w[:, :, :, np.newaxis], [2, 0, 1, 3])
    
    # Predict slice
    predictions = model_selection.predict(series_w)
    slice_label = get_slice_number_from_prediction(predictions)
    print("Predicted slice:", slice_label)
    
    return slice_label

if __name__ == '__main__':
    args = parse_args()
    meta = pd.read_csv(os.path.join(args.input, 'meta.csv'))
    
    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('.nii.gz')]
    # Print "Found n files" where n is the number of files with emoji
    print(f"📄 Found {len(filenames)} files")
    print()
    for i, filename in enumerate(filenames):
        print(f"[{i+1}/{len(filenames)}] Processing {filename}...")
        row = meta[meta['filename'] == filename]
        print(row)
        print(row['age'])
        age = row['age'].values[0]
        sex = row['sex'].values[0]
        filepath = os.path.join(args.input, filename)
        
        slice_label = predict_slice(
            age=age, 
            sex=sex, 
            input_path=filepath,
            model_weight_path_selection=args.model_weight_path_selection, 
            cuda_visible_devices=args.cuda_visible_devices
        )
        meta.loc[meta['filename'] == filename, 'slice_label'] = slice_label
        print()
    meta.to_csv(os.path.join(args.output, 'meta.csv'), index=False)
    print('✅ meta.csv saved with slice labels')