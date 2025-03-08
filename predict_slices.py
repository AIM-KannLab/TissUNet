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
from scripts.preprocess_utils import load_nii, enhance_noN4

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

# register the MRI to the template     
def register_to_template(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('golden_image/mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        #print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
        except:
            print("Cannot transform", rename_id)
            
# function to select the correct MRI template based on the age  
def select_template_based_on_age(age):
    # MNI templates 
    age_ranges = {"golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7.999},
                    "golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13.99999},
                    "golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}
    for golden_file_path, age_values in age_ranges.items():
        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
            #print(golden_file_path)
            return golden_file_path
        
def predict_slice(age = 9, 
                sex="M",
                input_path = 'data/t1_mris/nihm_reg/clamp_1193_v1_t1w.nii.gz',
                path_to = "data/temp/", 
                cuda_visible_devices="0",
                model_weight_path_selection = 'model_weights/densenet_itmt2.hdf5'):
    
    print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if len(physical_devices) == 0:
        physical_devices = tf.config.experimental.list_physical_devices('CPU')
    else:   
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
    # load model
    model_selection = DenseNet(img_dim=(256, 256, 1), 
                    nb_layers_per_block=16, nb_dense_block=4, growth_rate=16, nb_initial_filters=16, 
                    compression_rate=0.5, sigmoid_output_activation=True, 
                    activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
    model_selection.load_weights(model_weight_path_selection)
    print('Loaded:' ,model_weight_path_selection)  
    
    # Process single image
    patient_id = os.path.basename(input_path).split(".")[0]
    new_path_to = os.path.join(path_to, patient_id)
    os.makedirs(new_path_to, exist_ok=True)

    # Register image to MNI template
    golden_file_path = select_template_based_on_age(age)
    print("Registering to template:", golden_file_path)
    register_to_template(input_path, new_path_to, golden_file_path, "registered.nii.gz", create_subfolder=False)

    # Enhance and z-score normalize image
    no_z_folder = os.path.join(new_path_to, "no_z")
    os.makedirs(no_z_folder, exist_ok=True)
    
    image_sitk = sitk.ReadImage(os.path.join(new_path_to, "registered.nii.gz"))
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = enhance_noN4(image_array)
    sitk.WriteImage(sitk.GetImageFromArray(image_array), os.path.join(no_z_folder, "registered_no_z.nii"))

    # Run external z-score normalization
    subprocess.getoutput(f"zscore-normalize {os.path.join(no_z_folder, 'registered_no_z.nii')} -o {os.path.join(new_path_to, 'registered_z.nii')}")

    # Load normalized image
    image_sitk = sitk.ReadImage(os.path.join(new_path_to, "registered_z.nii"))
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
    meta = pd.read_csv(os.path.join(args.input, 'meta.csv'))
    
    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('.nii.gz')]
    # Print "Found n files" where n is the number of files with emoji
    print(f"📄 Found {len(filenames)} files")
    print()
    temp_path = os.path.join(args.input, 'temp')
    shutil.rmtree(temp_path, ignore_errors=True)
    os.makedirs(temp_path, exist_ok=True)
    for i, filename in enumerate(filenames):
        print(f"[{i+1}/{len(filenames)}] Processing {filename}...")
        row = meta[meta['filename'] == filename]
        age = row['age'].values[0]
        sex = row['sex'].values[0]
        filepath = os.path.join(args.input, filename)
        
        slice_label = predict_slice(
            age=age, 
            sex=sex, 
            input_path=filepath,
            path_to=temp_path,
            cuda_visible_devices=args.cuda_visible_devices,
            model_weight_path_selection=args.model_weight_path_selection, 
        )
        meta.loc[meta['filename'] == filename, 'slice_label'] = slice_label
        print()
    meta.to_csv(os.path.join(args.output, 'meta.csv'), index=False)
    print('✅ meta.csv saved with slice labels')