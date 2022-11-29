#!/bin/bash

# Download weights, move to data dir
wget https://pjreddie.com/media/files/yolov3.weights

if [ ! -d "./data" ]; then
    mkdir data
fi
mv yolov3.weights data

