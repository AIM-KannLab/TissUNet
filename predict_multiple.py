import argparse
import os

def predict_dataset(input_path, output_base):
    # Extract the dataset name
    dataset_name = os.path.basename(input_path)
    
    # Construct output path
    output_path = os.path.join(output_base, dataset_name)
    
    print(f"Predicting dataset:")
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
    
    # Run prediction
    command = f"nnUNetv2_predict -i {input_path} -o {output_path} -d 003 -c 3d_fullres -f all -device cuda"
    os.system(command)

def main():
    parser = argparse.ArgumentParser(description='Predict multiple datasets')
    parser.add_argument('--input_base', type=str, 
                        default='/media/sda/Elvira/TissUNet/extracranial/data/3d_inputs',
                        help='Base directory containing input datasets')
    parser.add_argument('--output_base', type=str,
                        default='/media/sda/Elvira/TissUNet/extracranial/data/3d_outputs',
                        help='Base directory for outputs')
    parser.add_argument('--datasets', nargs='+', help='List of dataset names to process')
    
    args = parser.parse_args()
    
    for dataset in args.datasets:
        input_path = os.path.join(args.input_base, dataset)
        print(f"\nProcessing: {dataset}")
        predict_dataset(input_path, args.output_base)
        print(f"Finished processing: {dataset}")

if __name__ == "__main__":
    main()