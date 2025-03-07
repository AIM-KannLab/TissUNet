# Preprocessing
import os
import argparse
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--input',  '-i', type=str, required=True,  help='Input directory with NIfTI files and meta.csv')
    parser.add_argument('--output', '-o', type=str, required=False, help='Output directory (optional)')
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

def reorient_to_lpi(file: nib.Nifti1Image):
    """
    Reorients a NIfTI file to LPI (Left-Posterior-Inferior) orientation.

    :param file: The NIfTI image to be reoriented.
    :type file: nib.Nifti1Image
    :return: The reoriented NIfTI image.
    :rtype: nib.Nifti1Image
    """
    original_ornt = axcodes2ornt(aff2axcodes(file.affine))
    target_ornt = axcodes2ornt(('L', 'P', 'I'))
    transform = ornt_transform(original_ornt, target_ornt)
    reoriented_file = file.as_reoriented(transform)
    return reoriented_file

def main(args):
    os.makedirs(args.output, exist_ok=True)
    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('.nii') or fn.endswith('.nii.gz')]
    
    meta = pd.read_csv(os.path.join(args.input, 'meta.csv'))
    print('🔎 Checking meta.csv')
    if not all([col in meta.columns for col in ['filename', 'age', 'sex']]):
        raise ValueError('meta.csv must contain filename, age and sex columns')
    for filename in filenames:
        if filename not in meta['filename'].values:
            raise ValueError(f'{samplename} not found in meta.csv')
    meta['filename'] = meta['filename'].apply(lambda x: x.replace('.nii.gz', '_0000.nii.gz').replace('.nii', '_0000.nii.gz'))
    meta.to_csv(os.path.join(args.output, 'meta.csv'), index=False)
    print('✅ meta.csv checked and saved')
    # Preprocessing filenames
    print('🔄 Preprocessing filenames')
    print()
    for i, file_name in enumerate(sorted(filenames)):
        file_path = os.path.join(args.input, file_name)
        print(f"[{i+1}/{len(filenames)}] Processing {file_path}...")
        file = read_nii_with_fix(file_path)
        if aff2axcodes(file.affine) != ('L', 'P', 'I'):
            print(f"\tFound orientation {aff2axcodes(file.affine)}. Reorienting to LPI...")
            file = reorient_to_lpi(file)
        else:
            print("\tAlready LPI")
        
        output_file_name = None
        if file_name.endswith('_0000.nii.gz'):
            output_file_name = file_name
        elif file_name.endswith('.nii.gz'):
            output_file_name = file_name.replace('.nii.gz', '_0000.nii.gz')
        elif file_name.endswith('.nii'):
            output_file_name = file_name.replace('.nii', '_0000.nii.gz')
        output_file_path = os.path.join(args.output, output_file_name)
        nib.save(file, output_file_path)
        print(f"\tSaved to {output_file_path}")
        print()
    print('🎉 Preprocessing Done!')

if __name__ == '__main__':
    args = parse_args()
    main(args)