import time
import argparse
import os
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

from detector.darknet import Darknet
from detector.utils import predict_transform, get_detections, resize_image, prep_image

def parse_args() -> argparse.Namespace:
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="YOLOv3 Detector")

    # data directories
    parser.add_argument("--images", dest="images", help="Directory containing input images",
        default="imgs", type=str)
    parser.add_argument("--detect", dest="detect", help="Directory to save images annotated with detections.",
        default="detects", type=str)
    
    # hyperparams
    parser.add_argument("--b_sz", dest="b_sz", help="Batch size", default=1, type=int)
    parser.add_argument("--obj_conf", dest="obj_conf", help="Object confidence threshold to detect an object", default=0.5, type=float)
    parser.add_argument("--nms_conf", dest="nms_conf", help="Non-maximum suppression threshold to filter out duplicate detections", default=0.4, type=float)

    # config files
    parser.add_argument("--cfg_file", dest="cfg_file", help="Path to YOLOv3 config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--wts_file", dest="wts_file", help="Path to pre-trained YOLOv3 weights file", default="data/yolov3.weights", type=str)

    # resolution
    parser.add_argument("--res", dest="res", help="Input image resolution. Higher -> more accurate, Lower -> faster.", default=416, type=int)

    return parser.parse_args()

def load_classes(name_file: Path):
    with open(str(name_file), "r") as f:
        names = [x[:-1] for x in f.readlines()]
    return names

if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    img_dir = Path(args.images)
    detect_dir = Path(args.detect)
    b_sz = args.b_sz
    obj_conf = args.obj_conf
    nms_conf = args.nms_conf
    start = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read in COCO class names
    name_file = Path("data/coco.names")
    names = load_classes(name_file)

    # initialize yolov3
    print("Initializing YOLOv3...")
    model = Darknet(args.cfg_file)
    model.modify_net_info(batch=b_sz, width=args.res, height=args.res)
    model.load_weights(args.wts_file)

    assert model.net_info["height"] % 32 == 0, "Resolution must be multiple of 32."
    assert model.net_info["height"] > 32, "Resolution must be greater than 32."
    model.eval() # eval mode
    print("Initialized YOLOv3.")

    # initialize output directory
    if not detect_dir.is_dir():
        detect_dir.mkdir(parents=True, exist_ok=True)

    # read, load input images
    read_dir_chkpt = time.time()
    img_list = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))

    load_img_chkpt = time.time()
    imgs = [cv2.imread(str(img_file)) for img_file in img_list]

    # prep images
    prepped_imgs = list(map(prep_image, imgs, [args.res for _ in range(len(imgs))]))
    img_dims = torch.tensor([(img.shape[1], img.shape[0]) for img in imgs], dtype=torch.float32).repeat(1, 2)

    # create batches
    leftover = len(imgs) % b_sz > 0
    if b_sz > 1:
        n_batches = (len(imgs) % b_sz) + int(leftover)
        prepped_imgs = [torch.cat(torch[i*b_sz:min((i+1)*b_sz, len(prepped_imgs))], dim=0) for i in range(n_batches)]
    