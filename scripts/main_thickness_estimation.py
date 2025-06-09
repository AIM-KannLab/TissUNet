# from mri_processing import process_mri_files
from mask_processing import load_and_binarize_mask, get_slice_for_file
from thickness_calculation import (
    find_longest_contour,
    calculate_checkpoints,
    get_slice_range,
    calculate_statistics,
    calculate_thickness,
    assign_checkpoints_to_quadrants,
    fill_holes_in_contours,
)
from utils import find_mask_files, save_file_results, save_global_results
from plotting import plot_results
import os
import numpy as np
import pandas as pd
import shutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import argparse

skipped_images_df = pd.DataFrame(columns=["filename", "reason"])


def process_mask_file(
    mask_path,
    lookup_slice_table,
    plot_output_dir,
    csv_output_dir,
    global_csv_results,
    dataset_name,
):
    """
    Process a single mask file and perform thickness calculations.

    Parameters:
        mask_path (str): Path to the binary mask file
        lookup_slice_table (str): Path to CSV file containing slice information
        plot_output_dir (str): Directory to save visualization plots
        csv_output_dir (str): Directory to save CSV results
        global_csv_results (list): List to store results for all processed files
        dataset_name (str): Name of the dataset being processed

    Returns:
        None

    Note:
        Skips processing if no matching slice label is found and adds the filename
        to skipped_images_df.
    """
    filename = os.path.basename(mask_path).split(".")[0]
    mask_binary = load_and_binarize_mask(mask_path)
    z_index = get_slice_for_file(mask_path, lookup_slice_table)

    if z_index is None:
        print(f"No matching slice label found for file: {filename}")
        return [], [{"filename": filename, "reason": "no slice label"}]

    filename_plot_dir = os.path.join(plot_output_dir, filename)
    os.makedirs(filename_plot_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    z_slice_range = get_slice_range(z_index, slice_range=15)
    csv_results = []
    skipped_images = []
    single_contour_count = 0

    for z in z_slice_range:
        success, reason = process_slice(
            mask_binary,
            z,
            filename,
            filename_plot_dir,
            csv_results,
            dataset_name,
        )
        if not success:
            # Record the skipped image with the reason
            skipped_images.append({"filename": filename, "reason": reason})
            return csv_results, skipped_images  # Skip for serious issues

    # Save results
    save_file_results(csv_results, csv_output_dir, filename)
    global_csv_results.extend(csv_results)
    return csv_results, skipped_images


def process_slice(
    mask_binary, z, filename, filename_plot_dir, csv_results, dataset_name, plot_results_enabled=False
):
    """
    Process a single slice for thickness calculations and visualization.

    Parameters:
        mask_binary (np.ndarray): 3D binary mask array
        z (int): Z-index of the slice to process
        filename (str): Name of the file being processed
        filename_plot_dir (str): Directory to save plots for this file
        csv_results (list): List to store results for this file
        dataset_name (str): Name of the dataset being processed

    Returns:
        bool: False if processing should be skipped, None otherwise

    The function performs the following steps:
    1. Extracts and rotates the specified slice
    2. Fills holes in contours
    3. Finds the longest contour
    4. Calculates checkpoints and thickness measurements
    5. Assigns measurements to quadrants
    6. Saves visualization plots
    """
    mask_slice = mask_binary[:, :, z - 1]
    # mask_slice = np.rot90(mask_slice)
    # Process contours and fill holes
    mask_slice = fill_holes_in_contours(mask_slice)
    result = find_longest_contour(mask_slice)
    if result is None:
        print(f"No contours found in slice {z} of image {filename}. Skipping slice.")
        return False, "no contours"  # Signal to skip the slice

    # Unpack results
    x_coords, y_coords, contours = result

    checkpoints = calculate_checkpoints(x_coords, y_coords, num_checkpoints=100)
    contour = np.column_stack((y_coords, x_coords))
    thicknesses = calculate_thickness(mask_slice, checkpoints, contour)

    center_y, center_x = mask_slice.shape[0] // 2, mask_slice.shape[1] // 2
    quadrant_data = assign_checkpoints_to_quadrants(
        checkpoints, thicknesses, center_x, center_y
    )

    # Validate quadrant_data before calculating statistics
    if any(len(values) == 0 for values in quadrant_data.values()):
        print(f"Empty quadrant data for slice {z} in image {filename}. Skipping image.")
        # shutil.rmtree(filename_plot_dir)
        return False, "broken"  # Signal to skip the image

    quadrant_stats = {
        quad: calculate_statistics(values) for quad, values in quadrant_data.items()
    }

    csv_results.extend(
        [
            {
                "dataset": dataset_name,
                "filename": filename,
                "z_index": z,
                "thickness": thickness,
                "quadrant": quad,
            }
            for quad, thickness_list in quadrant_data.items()
            for thickness in thickness_list
        ]
    )

    plot_save_path = os.path.join(filename_plot_dir, f"{filename}_slice_{z}.png")
    if plot_results_enabled:
        plot_results(
            mask_slice,
            checkpoints,
            thicknesses,
            quadrant_stats,
            center_x,
            center_y,
            contour,
            z,
            filename,
            save_path=plot_save_path,
        )
    return True, None


def save_global_results(results, output_dir):
    """
    Save global results to CSV with append mode.

    Parameters:
        results (list): List of dictionaries containing thickness measurements
        output_dir (str): Directory to save the CSV file

    Returns:
        None

    Notes:
        - Creates 'global_thickness_calculation.csv' in the output directory
        - Appends to existing file if it exists, creates new file with header if it doesn't
        - Skips if results list is empty
    """
    if not results:  # If results is empty, return
        return

    output_file = os.path.join(output_dir, "global_thickness_calculation.csv")

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # If file exists, append without header. If not, write with header
    df.to_csv(output_file, header=True, index=False)

    print(f"Saved {len(results)} results to {output_file}")


# Define this function at module level
def process_file_wrapper(
    file,
    processed_image_dir,
    lookup_slice_table,
    plot_output_dir,
    csv_output_dir,
    dataset_name,
):
    mask_path = os.path.join(processed_image_dir, file)
    local_results = []
    try:
        local_results, skipped_images = process_mask_file(
            mask_path,
            lookup_slice_table,
            plot_output_dir,
            csv_output_dir,
            local_results,
            dataset_name,
        )
        print(f"Successfully processed {file}")
        return local_results, skipped_images, None
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return None, None, (file, str(e))

def filter_outliers(df):
    # Calculate the 99.5th and 0.5th percentiles
    threshold_upper = df['thickness'].quantile(0.995)
    threshold_lower = df['thickness'].quantile(0.005)
    
    # Filter out rows where thickness is outside the thresholds
    df_filtered = df[(df['thickness'] >= threshold_lower) & (df['thickness'] <= threshold_upper)]
    
    return df_filtered

def aggregate_thickness(data, method="median"):
    if method not in ["median", "mean"]:
        raise ValueError("Invalid method. Choose 'median' or 'mean'.")

    # Aggregate thickness by filename
    if method == "median":
        thickness_agg = data.groupby("filename", as_index=False)["thickness"].median()
    else:
        thickness_agg = data.groupby("filename", as_index=False)["thickness"].mean()

    # Select all other columns except thickness and keep one row per filename
    other_data = data.drop(columns=["thickness"]).drop_duplicates(subset="filename")

    # Merge the aggregated thickness back to the other data
    result = pd.merge(other_data, thickness_agg, on="filename", how="left")

    return result

def main():
    """
    Main execution function for the thickness calculation pipeline.

    The pipeline consists of the following steps:
    1. Configure paths based on dataset name
    2. Process MRI files and convert to appropriate format
    3. Process mask files for each subject:
        - Calculate thickness measurements
        - Generate visualization plots
        - Save results to CSV files
    4. Save global results and list of skipped images

    Dataset naming convention:
    - Expected format: "{base_name}-MRI" (e.g., "CERMEP-MRI" or "SynthRad-MRI")
    - Base name is used to locate corresponding slice information CSV
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Skull thickness estimation pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help='Dataset name (e.g., "calgary_super")',
    )
    parser.add_argument(
        "--lookup-slice-table",
        type=str,
        help="Path to lookup slice table CSV file. Default: /media/sda/Elvira/extracranial/data/supp_data/metadata_{dataset}.csv",
    )
    parser.add_argument(
        "--csv-output-dir",
        type=str,
        help="Directory for CSV output. Default: /media/sda/Elvira/extracranial/results/results_thickness/{dataset}",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        help="Directory for plot output. Default: /media/sda/Elvira/extracranial/results/plots/{dataset}",
    )
    parser.add_argument(
        "--processed-image-dir",
        type=str,
        help="Directory for processed images. Default: /media/sda/Elvira/extracranial/data/3d_outputs/{dataset}",
    )
    args = parser.parse_args()

    start_time = time.time()
    # Dataset configuration
    dataset_name = args.dataset
    # Split the dataset name into base name and modality
    # base_name = dataset_name.split("_")[0]

    # Base directories
    current_dir = os.getcwd()

    # Default path configurations based on dataset name
    default_paths = {
        "lookup_slice_table": f"/media/sda/Elvira/extracranial/data/supp_data/metadata_{dataset_name}.csv",
        "csv_output_dir": f"/media/sda/Elvira/extracranial/results/results_thickness/{dataset_name}",
        "plot_output_dir": f"/media/sda/Elvira/extracranial/results/plots/{dataset_name}",
        "processed_image_dir": f"/media/sda/Elvira/extracranial/data/3d_outputs/{dataset_name}",
    }

    # Use command line arguments if provided, otherwise use defaults
    paths = {
        "lookup_slice_table": args.lookup_slice_table
        or default_paths["lookup_slice_table"],
        "csv_output_dir": args.csv_output_dir or default_paths["csv_output_dir"],
        "plot_output_dir": args.plot_output_dir or default_paths["plot_output_dir"],
        "processed_image_dir": args.processed_image_dir
        or default_paths["processed_image_dir"],
    }

    # Print path information for user reference
    print("Using the following paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    skipped_images_path = os.path.join(paths["csv_output_dir"], "skipped_images.csv")
    skipped_images_df = pd.DataFrame(columns=["filename", "reason"])
    lookup_table = pd.read_csv(paths["lookup_slice_table"])
    global_csv_results = []

    # Ensure necessary directories exist
    for dir_path in [
        paths["csv_output_dir"],
        paths["plot_output_dir"],
        paths["processed_image_dir"],
    ]:
        os.makedirs(dir_path, exist_ok=True)

    nii_files = [
        f for f in os.listdir(paths["processed_image_dir"]) if f.endswith(".nii.gz")
    ]
    total_files = len(nii_files)

    print(f"Found {total_files} .nii.gz files to process")

    # Parallel processing
    num_workers = max(
        1, multiprocessing.cpu_count() - 2
    )  # Leave one core free, but ensure at least 1 worker
    # num_workers = 50
    print(f"Using {num_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Execute tasks in parallel and collect results
        futures = [
            executor.submit(
                process_file_wrapper,
                file,
                paths["processed_image_dir"],
                paths["lookup_slice_table"],
                paths["plot_output_dir"],
                paths["csv_output_dir"],
                dataset_name,
            )
            for file in nii_files
        ]

    # Collect results
    processed_files = set()  # Track unique processed files
    skipped_files = set()  # Track unique skipped files
    error_files = set()  # Track unique error files

    for future in futures:
        result, skipped, error = future.result()
        if result:
            # 1 Convert your list of row-dicts into a DataFrame
            df_result = pd.DataFrame(result)

            # 2 Filter out the outliers by thickness
            df_filtered = filter_outliers(df_result)

            # 3 Aggregate to mean thickness per filename
            df_mean = aggregate_thickness(df_filtered, method="mean")

            # 4 Append each aggregate row (as a dict) into your global list
            for _, row in df_mean.iterrows():
                global_csv_results.append({
                    "dataset":   dataset_name,
                    "filename":  row["filename"],
                    "z_index":   -1,             # placeholder since it’s an aggregate
                    "thickness": row["thickness"],
                    "quadrant":  "mean"
                })

            # 5 Mark this file as processed
            processed_files.add(df_mean["filename"].iloc[0])

        if skipped:
            # Add any skipped images from this process to DataFrame
            for item in skipped:
                skipped_images_df = pd.concat(
                    [skipped_images_df, pd.DataFrame([item])], ignore_index=True
                )
                if "filename" in item:
                    skipped_files.add(item["filename"])

        if error:
            file, reason = error
            skipped_images_df = pd.concat(
                [
                    skipped_images_df,
                    pd.DataFrame([{"filename": file, "reason": reason}]),
                ],
                ignore_index=True,
            )
            error_files.add(file)

    # Account for cases where a file might appear in multiple categories
    # (prioritizing successful processing)
    final_skipped_files = skipped_files - processed_files
    final_error_files = error_files - processed_files - final_skipped_files

    print("\nProcessing complete!")
    print(f"Total files examined: {total_files}")
    # print(f"Files with successful measurements: {len(processed_files)}")
    print(f"Files skipped due to issues: {len(final_skipped_files)}")
    print(f"Files with processing errors: {len(final_error_files)}")
    print(
        f"Total accounted for: {len(processed_files) + len(final_skipped_files) + len(final_error_files)}"
    )
    print(f"Total entries in skipped_images.csv: {len(skipped_images_df)}")

    skipped_images_df.to_csv(skipped_images_path, index=False)
    print(f"Skipped images saved to {skipped_images_path}")

    # Save all results at the end - don't do this if N of subjects >650
    save_global_results(global_csv_results, paths["csv_output_dir"])

    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"\nTotal execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()
