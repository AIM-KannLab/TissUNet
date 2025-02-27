# Setup
You can use this pip example or conda if you want. Just make sure to install `python3.11` and all necessary dependancies.
```
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

# Download Weights
You can download weights of TissUNet using [this link](https://drive.google.com/drive/folders/18c06FU825eIsgyscO1CEbZf8jszzKJdT?usp=drive_link)

# Project Structure
```
TissUNet \ <cloned repo>
    venv \
    mr \
    nnUNet_results \
        Dataset003_synthrad \
            nnUNetTrainer__nnUNetPlans__3d_fullres \
    .gitignore
    README.md
    preprocess.py
    postprocess.py
    compute_metrics.py
    requirements.in
    requirements.txt
```

# Preprocess
The following script will reorient all `.nii.gz` in `<in_dir>` into LPI orientation and add `_0000.nii.gz` postfix. If `<out_dir>` is not specified it will overwrite files in `<in_dir>`.
```
python preprocess.py -i <in_dir> [-o <out_dir>]
```
Example:
```
python preprocess.py -i mr -o mr_pre
```

# Predict
This will run TissUNet on all `.nii.gz` files in `<in_dir>` and write results in `<out_dir>`.
```
export nnUNet_raw="$(pwd)/<any_path_really_this_stuff_is_required_even_though_not_used>"
export nnUNet_preprocessed="$(pwd)/<any_path_really_this_stuff_is_not_used_but_suppresses_the_warning>"
export nnUNet_results="$(pwd)/<relative_path_to_nnUNet_results>"
nnUNetv2_predict -i <in_dir> \
                 -o <out_dir> \
                 -d 003 -c 3d_fullres -f all -device <gpu/cpu>
```
Example:
```
export nnUNet_raw="$(pwd)/nnUNet_raw" # This path does not exist lol
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"
nnUNetv2_predict -i mr_pre \
                 -o preds \
                 -d 003 -c 3d_fullres -f all -device gpu
```

# Post-process
The following script will filter brain mask (retain only the largest connected componnent) and deface if required
```
python postprocess.py -mi <mr_input_path> \
                      -pi <preds_input_path> \
                      -mo <mr_output_path> \
                      -po <preds_output_path> \
                      --deface
```
Example:
```
python postprocess.py -mi mr_pre \
                      -pi preds \
                      -mo mr_post \
                      -po preds_post \

python postprocess.py -mi mr_pre \
                      -pi preds \
                      -mo mr_post_def \
                      -po preds_post_def \
                      --deface
```

# Computation of metrics
To compute metrics for a single directory of predictions use:
```
python compute_metrics.py -pi <preds_input_path> \
                          -mo <metrics_json_output_file_path>
```
Example:
```
python compute_metrics.py -pi preds \
                          -mo preds/metrics.csv
```

# Known Issues
- For some slices IMEA throws a warning during 2D (micro) metrics computation: `Slope is zero slope --> fractal dimension will be set to zero`.
- For some slices the volumetrics computed by IMEA and by hand differ by a few pixels.