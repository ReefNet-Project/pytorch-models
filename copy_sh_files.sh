#!/bin/bash

# List of files to copy
files=("run_beit.sh" "run_convnext.sh" "run_deit.sh" "run_effecientnet.sh" "run_resnet.sh" "run_swin.sh" "run_vit.sh")

root_folder="/home/yahiab/reefnet_project/yahia_code/transformer_based_baselines/pytorch-image-models/job_scripts/new_scripts/cross-source/07_filtration/all"


# List of destination folders
folders=(
    "/home/yahiab/reefnet_project/yahia_code/transformer_based_baselines/pytorch-image-models/job_scripts/new_scripts/in-source/07_filtration/all"
)

# Loop through each folder and copy each file into it
for dir in "${folders[@]}"; do
    for file in "${files[@]}"; do
        cp "$root_folder/$file" "$dir/"
    done
done
