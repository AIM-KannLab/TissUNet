# Preprocessing
import argparse
import os
import shutil
import sys

import itk
import nibabel as nib
import pandas as pd
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--input',  '-i', type=str, required=True,  help='Input directory with NIfTI files and meta.csv')
    parser.add_argument('--output', '-o', type=str, required=False, help='Output directory (optional)')
    parser.add_argument('--no-register', action='store_false', dest='register', help='Disable MRI registration')
    args = parser.parse_args()
    if not args.output:
        args.output = args.input
    return args

def read_nii_with_fix(file_path: str):
    """
    Attempts to load a NIfTI file using nibabel. If loading fails, tries to fix it using SimpleITK.

    :param file_path: Path to the NIfTI file.
    :type file_path: str
    :return: Loaded NIfTI file if successful, otherwise None.
    :rtype: nib.Nifti1Image or None
    :raises Exception: If the file cannot be loaded or fixed.
    """
    file = None
    try:
        file = nib.load(file_path)
    except Exception as e:
        print(f'\033[91mError loading {file_path}: {e}\033[0m')
        print("Trying to fix the currupted file...")
        try:
            sitk_img = sitk.ReadImage(file_path)
            sitk.WriteImage(sitk_img, file_path)
            file = nib.load(file_path)
            print("Fixed the corrupted file succesfully!")
        except Exception as e:
            print(f'\033[91mError loading {file_path}: {e}\033[0m')
            print()
    return file
            
def check_meta_columns(meta):
    if not all([col in meta.columns for col in ['filename', 'age', 'sex']]):
        raise ValueError('meta.csv must contain filename, age and sex columns')
    
def get_nnunet_filename(file_name):
    output_file_name = None
    if file_name.endswith('_0000.nii.gz'):
        output_file_name = file_name
    elif file_name.endswith('.nii.gz'):
        output_file_name = file_name.replace('.nii.gz', '_0000.nii.gz')
    elif file_name.endswith('.nii'):
        output_file_name = file_name.replace('.nii', '_0000.nii.gz')
    return output_file_name

# function to select the correct MRI template based on the age  
def select_template_based_on_age(age):
    # MNI templates 
    #  gestational ages 36 weeks to 44 weeks (0-1 month) are included in the repo but not tested
    age_ranges = {  ### 0 to 5 years (60 months)
                    "golden_image/mni_templates/months/00Month/BCP-00M-T1.nii.gz":{"min_age":0,   "max_age":0.99999/12}, # 0-1 month
                    "golden_image/mni_templates/months/01Month/BCP-01M-T1.nii.gz":{"min_age":1/12,   "max_age":1.99999/12}, # 1-2 months
                    "golden_image/mni_templates/months/02Month/BCP-02M-T1.nii.gz":{"min_age":2/12,   "max_age":2.99999/12}, # 2-3 months
                    "golden_image/mni_templates/months/03Month/BCP-03M-T1.nii.gz":{"min_age":3/12,   "max_age":3.99999/12}, # 3-4 months
                    "golden_image/mni_templates/months/04Month/BCP-04M-T1.nii.gz":{"min_age":4/12,   "max_age":4.99999/12}, # 4-5 months
                    "golden_image/mni_templates/months/05Month/BCP-05M-T1.nii.gz":{"min_age":5/12,   "max_age":5.99999/12}, # 5-6 months
                    "golden_image/mni_templates/months/06Month/BCP-06M-T1.nii.gz":{"min_age":6/12,   "max_age":6.99999/12}, # 6-7 months
                    "golden_image/mni_templates/months/07Month/BCP-07M-T1.nii.gz":{"min_age":7/12,   "max_age":7.99999/12}, # 7-8 months
                    "golden_image/mni_templates/months/08Month/BCP-08M-T1.nii.gz":{"min_age":8/12,   "max_age":8.99999/12}, # 8-9 months
                    "golden_image/mni_templates/months/09Month/BCP-09M-T1.nii.gz":{"min_age":9/12,   "max_age":9.99999/12}, # 9-10 months
                    "golden_image/mni_templates/months/10Month/BCP-10M-T1.nii.gz":{"min_age":10/12,  "max_age":10.99999/12}, # 10-11 months
                    "golden_image/mni_templates/months/11Month/BCP-11M-T1.nii.gz":{"min_age":11/12,  "max_age":11.99999/12}, # 11-12 months
                    "golden_image/mni_templates/months/15Month/BCP-15M-T1.nii.gz":{"min_age":12/12,  "max_age":15.999999/12}, # 12-15 months (1-1.25 years)
                    "golden_image/mni_templates/months/18Month/BCP-18M-T1.nii.gz":{"min_age":16/12,  "max_age":18.9999/12}, 
                    "golden_image/mni_templates/months/18Month/BCP-21M-T1.nii.gz":{"min_age":19/12,  "max_age":21.9999/12}, # 18-21 months (1.5-1.75 years)
                    "golden_image/mni_templates/months/24Month/BCP-24M-T1.nii.gz":{"min_age":22/12,  "max_age":24.9999/12}, # 2-2.5 years
                    "golden_image/mni_templates/months/36Month/BCP-36M-T1.nii.gz":{"min_age":25/12,  "max_age":36.9999/12}, # 2.5-3 years
                    "golden_image/mni_templates/months/48Month/BCP-48M-T1.nii.gz":{"min_age":37/12,  "max_age":48.9999/12}, # 3-4 years
                    "golden_image/mni_templates/months/60Month/BCP-60M-T1.nii.gz":{"min_age":49/12,  "max_age":60.9999/12}, # 4-5 years
                    ### 5 to 150 years
                    "golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii": {"min_age":5,  "max_age":7.999},
                    "golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8,  "max_age":13.99999},
                    "golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":150}}
    

    for golden_file_path, age_values in age_ranges.items():
        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
            return golden_file_path
   
# register the MRI to the template     
def register_to_template(input_image_path, output_image_path, fixed_image_path):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('golden_image/mni_templates/Parameters_Rigid.txt')
    
    moving_image = itk.imread(input_image_path, itk.F)
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False)
    itk.imwrite(result_image, output_image_path)

def main(args):
    shutil.rmtree(args.output, ignore_errors=True)
    os.makedirs(args.output, exist_ok=True)
    # Process Meta
    meta = pd.read_csv(os.path.join(args.input, 'meta.csv'))
    print('🔎 Checking meta.csv')
    check_meta_columns(meta=meta)    
    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('.nii') or fn.endswith('.nii.gz')]
    for filename in filenames:
        if filename not in meta['filename'].values:
            print(f'The filename {filename} is not found in meta.csv. Please, make sure that all filenames from input directory are present in meta.csv')
            sys.exit(1)
    meta['filename'] = meta['filename'].apply(lambda x: get_nnunet_filename(x))
    meta.to_csv(os.path.join(args.output, 'meta.csv'), index=False)
    print('✅ meta.csv checked and saved')
    # Process NII
    print('🔄 Preprocessing filenames\n')
    for i, file_name in enumerate(sorted(filenames)):
        file_path = os.path.join(args.input, file_name)
        print(f"[{i+1}/{len(filenames)}] Processing {file_path}...")
        file = read_nii_with_fix(file_path)

        output_file_name = get_nnunet_filename(file_name)
        output_file_path = os.path.join(args.output, output_file_name)
        nib.save(file, output_file_path)
        
        # Register to the template
        if args.register:
            print(f"\tRegistering to the template...")
            age = meta[meta['filename'] == output_file_name]['age'].values[0]
            template_path = select_template_based_on_age(age)
            register_to_template(output_file_path, output_file_path, template_path)
        
        print(f"\tSaved to {output_file_path}")
        print()
    print('🎉 Preprocessing Done!')

if __name__ == '__main__':
    args = parse_args()
    main(args)