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
from detector.utils import predict_transform, get_detections, resize_image, prep_image, draw_bbox

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

    # load colors
    with open("data/pallete", "rb") as f:
        colors = pickle.load(f)

    # initialize yolov3
    print("Initializing YOLOv3...")
    model = Darknet(args.cfg_file, device=device)
    input_res = int(model.net_info["height"])
    model.modify_net_info(batch=b_sz, width=args.res, height=args.res)
    model.load_weights(args.wts_file)
    print(model.device)

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
    prepped_imgs = list(map(prep_image, imgs, [args.res for _ in range(len(imgs))], [device for _ in range(len(imgs))]))
    img_dims = torch.tensor([(img.shape[1], img.shape[0]) for img in imgs], dtype=torch.float32).repeat(1, 2)

    # create batches
    leftover = len(imgs) % b_sz > 0
    if b_sz > 1:
        n_batches = (len(imgs) % b_sz) + int(leftover)
        prepped_imgs = [torch.cat(torch[i*b_sz:min((i+1)*b_sz, len(prepped_imgs))], dim=0).to(device) for i in range(n_batches)]

    # Get detections on each batch
    output = None
    start_detect_loop = time.time()
    for i, batch in enumerate(prepped_imgs):
        # Load the image
        pred_start = time.time()

        # Get model preds
        with torch.inference_mode():
            preds = model(batch).cpu()
        
        # Interpret detections from preds
        preds = get_detections(preds, args.obj_conf, nms_conf=args.nms_conf)
        pred_end = time.time()
        pred_time = (pred_end - pred_start)/b_sz

        # Print predictions for each batch image
        if isinstance(preds, torch.Tensor):
            for b_idx, image in enumerate(img_list[i*b_sz:min((i+1)*b_sz, len(img_list))]):
                print(f"Image: {str(image).split('/')[-1]}")
                print(f"Prediction time (sec): {pred_time:6.3f}")

                # Get object names
                objs = [names[int(x[-1])] for x in preds if int(x[0]) == b_idx]
                print(f"Objects detected: {', '.join(objs)}")
                print()
            
            # Convert b_idxs to img_idxs
            preds[:, 0] += i*b_sz

            # Concatenate
            if output !=  None:
                output = torch.cat((output, preds), dim=0)
            else:
                output = preds

        else:
            for b_idx, image in enumerate(img_list[i*b_sz:min((i+1)*b_sz, len(img_list))]):
                print(f"Image: {str(image).split('/')[-1]}")
                print(f"Prediction time (sec): {pred_time:6.3f}")
                print("No objects detected.")

        # Synchronize CUDA w/ CPU
        if str(device) == "cuda":
            print("Synchronized")
            torch.cuda.synchronize()

    # Draw bounding boxes
    if output is None:
        print("No objects detected.")
        exit()

    # get image with detections
    img_dims = torch.index_select(img_dims, dim=0, index=output[:, 0].long())

    # re-scale bbox edges to original image dimension
    scale = torch.min(args.res/img_dims, 1)[0].view(-1, 1)
    output[:, 1:5] /= scale

    # clip bounding boxes to original image dimensions
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dims[i, 0]) # left/right
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dims[i, 1]) # top/bottom

    # draw bounding boxes on images
    imgs = list(map(lambda x: draw_bbox(x, imgs, colors, names), output))

    # Save the images w/ bounding boxes drawn
    for img, img_path in zip(imgs, img_list):
        cv2.imwrite(f"{args.detect}/detect_{str(img_path).rsplit('/', 1)[-1]}", img)
    end = time.time()