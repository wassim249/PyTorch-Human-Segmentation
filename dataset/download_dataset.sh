#!/bin/bash

download_dataset() {
    # Download the dataset from GitHub
    git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git
    mv Human-Segmentation-Dataset-master data
    rm -rf data/.git
}

# Call the download_dataset function
download_dataset
