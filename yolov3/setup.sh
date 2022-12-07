#!/bin/bash

# Install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
rm Anaconda3-2022.10-Linux-x86_64.sh

# Get weights
bash get_weights.sh

# Get COCO names
bash get_coco.sh

# Set up git
git config --global user.name "dchung117"
git config --global user.email "dchung0330@gmail.com"