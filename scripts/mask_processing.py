import numpy as np
import nibabel as nib
import os
import pandas as pd


def load_and_binarize_mask(mask_path):
    """
    Load a .nii.gz file, extract its data, and binarize the mask.

    Parameters:
        mask_path (str): Path to the .nii.gz file.

    Returns:
        np.ndarray: Binarized mask array (1 for object, 0 for background).
        np.ndarray: Unique values in the original mask data.
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    #mask_data = np.flip(mask_data, axis=0)
    mask_data = np.flip(mask_data, axis=2)  # Flip superior-inferior orientation
    mask_data = np.rot90(mask_data, k=1)   
    # New orientation: LPI
    # mask_data = np.rot90(mask_data, k=3)   
    binarized_mask = np.where(mask_data == 4, 1, 0)
    return binarized_mask


def load_and_binarize_mask_ct(mask_path, threshold=300):
    """
    Load a .nii.gz file, extract its data, and binarize the mask.

    Parameters:
        mask_path (str): Path to the .nii.gz file.
        threshold (int, optional): Threshold value for binarization. Defaults to 300.

    Returns:
        np.ndarray: Binarized mask array (1 for object, 0 for background).
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    # mask_data = np.flip(mask_data, axis=0)
    binarized_mask = np.where(mask_data >= threshold, 1, 0)
    return binarized_mask


def get_slice_for_file(mask_path, csv_path):
    """
    Lookup the slice label for a given mask file using a CSV file as a reference.

    Parameters:
        mask_path (str): Path to the mask file.
        csv_path (str): Path to the CSV file containing file names and slice labels.

    Returns:
        int: The slice label corresponding to the given file, or None if not found.
    """
    df = pd.read_csv(csv_path, dtype={"ID": str})
    #df = pd.read_csv(csv_path)
    file_id = os.path.basename(mask_path).split(".")[0]

    # Convert both the lookup ID and file_id to strings and strip whitespace
    df["ID"] = df["ID"].astype(str).str.strip()
    file_id = str(file_id).strip()

    matching_row = df[df["ID"] == file_id]
    if not matching_row.empty:
        return matching_row["Slice label"].iloc[0]
    else:
        return None


def get_slice_for_file_ct(mask_path, lookup_table_path):
    """
    Get the slice index for a given file from the lookup table.
    """
    # Get file_name without extension and remove _ct suffix
    file_name = os.path.basename(mask_path).split(".")[0].replace("_ct", "")

    # Read the lookup table
    df = pd.read_csv(lookup_table_path)

    # Convert both the lookup ID and file_name to strings and strip whitespace
    df["ID"] = df["ID"].astype(str).str.strip()
    file_name = str(file_name).strip()

    # Find matching row
    matching_row = df[df["ID"] == file_name]

    if not matching_row.empty:
        return matching_row["Slice label"].iloc[0]
    else:
        return None
