#!/bin/bash

# Download coco dataset object names
wget https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names

# Move data
if [[ ! -d "./data" ]]; then
    mkdir data
fi
mv coco.names data