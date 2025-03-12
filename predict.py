import os
import argparse
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform
from typing import Tuple
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Predicting')
    parser.add_argument('--input',  '-i', type=str, required=True,  help='Input directory with MR images and meta.csv')
    parser.add_argument('--output', '-o', type=str, required=False, help='Output path for meta.csv')
    parser.add_argument('--device', '-d', type=str, required=True,  help='Device to run prediction on')
    parser.add_argument('--cleanup', action='store_true', help='Remove temporary directories after execution')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise ValueError('Input path does not exist')
    if not args.output:
        args.output = os.path.dirname(os.path.normpath(args.input))
    if args.device not in ['cpu', 'cuda']:
        raise ValueError('Device must be either cpu or gpu')
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

def reorient(file: nib.Nifti1Image, target_orientation: Tuple[str, str, str] = ('L', 'P', 'I')):
    """
    Reorients a NIfTI file to LPI (Left-Posterior-Inferior) orientation.

    :param file: The NIfTI image to be reoriented.
    :type file: nib.Nifti1Image
    :return: The reoriented NIfTI image.
    :rtype: nib.Nifti1Image
    """
    original_ornt = axcodes2ornt(aff2axcodes(file.affine))
    target_ornt = axcodes2ornt(target_orientation)
    transform = ornt_transform(original_ornt, target_ornt)
    reoriented_file = file.as_reoriented(transform)
    return reoriented_file

def get_orientation(file: nib.Nifti1Image):
    """
    Get the orientation of a NIfTI image based on its affine transformation.

    :param file: NIfTI image object.
    :type file: nib.Nifti1Image
    :return: Orientation codes representing the anatomical labels of the image axes.
    :rtype: tuple[str, str, str]
    """
    return nib.aff2axcodes(file.affine)

def main(args):
    temp_mri_path = os.path.join(args.input, 'temp')
    os.makedirs(temp_mri_path, exist_ok=True)
    
    # Remember orientations for each file that ends with _0000.nii.gz
    orientations = {}
    filenames = [fn for fn in os.listdir(args.input) if fn.endswith('_0000.nii.gz')]
    for filename in sorted(filenames):
        sample_name = filename.replace('_0000.nii.gz', '')
        file_path = os.path.join(args.input, filename)
        assert os.path.exists(file_path)
        file = read_nii_with_fix(file_path)
        
        orientations[sample_name] = get_orientation(file)
        reoriented_file = reorient(file, target_orientation=('L', 'P', 'I'))
        reoriented_path = os.path.join(temp_mri_path, filename)
        nib.save(reoriented_file, reoriented_path)
    
    # Run prediction script
    command = f"nnUNetv2_predict -i {temp_mri_path} -o {args.output} -d 003 -c 3d_fullres -f all -device {args.device}"
    subprocess.run(command, shell=True, check=True)
    
    # Reorient predictions back to original orientation
    for filename in sorted(filenames):
        sample_name = filename.replace('_0000.nii.gz', '')
        file_path = os.path.join(args.output, sample_name+'.nii.gz')
        assert os.path.exists(file_path)
        file = read_nii_with_fix(file_path)

        original_orientation = orientations[sample_name]
        file = reorient(file, target_orientation=original_orientation)
        nib.save(file, file_path)
        
    # Clean up
    if args.cleanup:
        os.rmdir(temp_mri_path)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)