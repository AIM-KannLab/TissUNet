# Batch processing for the skull estimation
To batch process multiple datasets for the skull thickness estimation, please add the folder names of the datasets you'd like to process under `datasets`. Our folder setup is as follows:

```
data/
├── 3d_outputs/
│   └── ExampleDataset/
│       ├── file1.nii.gz
│       ├── file2.nii.gz
│       └── file3.nii.gz
├── supp_data/
│   └── metadata_ExampleDataset.csv
results/
├── plots/
│   └── ExampleDataset/
└── results_thickness/
    └── ExampleDataset/
```

```
bash process_thickness_estimation.sh \
--lookup-slice-table "supp_data/metadata_$dataset.csv" \
--csv-output-dir "results/results_thickness/$dataset" \
--plot-output-dir "results/results_thickness/$dataset" \
--processed-image-dir "data/3d_outputs/$dataset" \

```
