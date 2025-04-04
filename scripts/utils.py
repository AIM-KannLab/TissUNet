import os
import pandas as pd
import glob


def find_mask_files(subfolder_path):
    """
    Locate mask files in a given subfolder.
    """
    mask_path_pattern = os.path.join(
        subfolder_path,
        "*.nii.gz",
    )
    return glob.glob(mask_path_pattern)


def find_mask_files_ct(subfolder_path):
    """
    Locate mask files in a given subfolder.
    """
    mask_path_pattern = os.path.join(
        subfolder_path,
        "*.nii.gz",
    )
    return glob.glob(mask_path_pattern)


def save_file_results(csv_results, csv_output_dir, filename):
    """
    Save results for a single file to a CSV.
    """
    results_df = pd.DataFrame(csv_results)
    file_output_path = os.path.join(
        csv_output_dir, f"{filename}_thickness_calculation.csv"
    )
    results_df.to_csv(file_output_path, index=False)
    print(f"Results for {filename} saved to {file_output_path}")


def save_global_results(global_csv_results, csv_output_dir):
    """
    Save all results to a global CSV.
    """
    global_results_df = pd.DataFrame(global_csv_results)
    global_output_path = os.path.join(
        csv_output_dir, "global_thickness_calculation.csv"
    )
    global_results_df.to_csv(global_output_path, index=False)
    print(f"Global results saved to {global_output_path}")
