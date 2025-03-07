import argparse
import os
from collections import Counter
import csv
from datetime import datetime

def count_files_by_suffix(directory):
    suffix_count = Counter()
    if os.path.exists(directory):
        for file in os.listdir(directory):
            suffix = os.path.splitext(file)[1]
            if suffix:  # Only count files with extensions
                suffix_count[suffix] += 1
    return dict(suffix_count)


def process_dataset(input_path):
    # Count files before processing
    input_files = count_files_by_suffix(input_path)
    print(f"\nInput directory contents:")
    for suffix, count in input_files.items():
        print(f"  {suffix}: {count} files")

    # Create CSV path
    csv_path = "/media/sda/Elvira/extracranial/file_counts.csv"
        
    # Extract the dataset name from the input path and remove '_reg'
    dataset_name = os.path.basename(input_path).split('_')[0]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Write to CSV before processing
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['Timestamp', 'Dataset', 'Directory', 'Suffix', 'Count'])
        
        for suffix, count in input_files.items():
            writer.writerow([timestamp, dataset_name, 'input', suffix, count])

    
    # Construct output path with fixed base directory
    output_base = "/media/sda/Elvira/extracranial/data/3d_inputs"
    output_path = os.path.join(output_base, dataset_name)
    
    print(f"Processing dataset:")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Try to create directory with verbose error checking
    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Directory created/verified: {output_path}")
        # Verify the directory exists and is writable
        if not os.path.exists(output_path):
            print(f"Error: Failed to create directory: {output_path}")
            return False
        if not os.access(output_path, os.W_OK):
            print(f"Error: No write permission for: {output_path}")
            return False
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False
    
    # Run preprocessing with more verbose output
    command = f"python preprocess.py --input {input_path} --output {output_path}"
    print(f"Executing command: {command}")
    return_code = os.system(command)
    if return_code != 0:
        print(f"Error: preprocessing failed for {input_path} with return code {return_code}")
        return False
    
    # Verify that something was actually created
    if not os.listdir(output_path):
        print(f"Warning: Output directory is empty after processing: {output_path}")
        return False
    # Count output files after processing
    output_files = count_files_by_suffix(output_path)
    print(f"\nOutput directory contents:")
    for suffix, count in output_files.items():
        print(f"  {suffix}: {count} files")

    # Append output file counts to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for suffix, count in output_files.items():
            writer.writerow([timestamp, dataset_name, 'output', suffix, count])
            
        
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess multiple datasets')
    parser.add_argument('--input_dirs', nargs='+', help='List of input directories to process')
    
    args = parser.parse_args()
    
    for input_dir in args.input_dirs:
        print(f"\nProcessing: {input_dir}")
        process_dataset(input_dir)
        print(f"Finished processing: {input_dir}")

if __name__ == "__main__":
    main()