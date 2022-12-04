from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn as nn

def predict_transform(preds: torch.Tensor, input_dim: int, anchors: list[tuple], n_classes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    # Get batch size, stride, grid size, bbox_attributes num_anchors
    b_sz = preds.shape[0]
    stride = input_dim // preds.shape[2]
    grid_size = input_dim // stride
    n_attrs = 5 + n_classes # object classification + bounding box transforms
    n_anchors = len(anchors)

    # Reshape predictions tensor
    preds = preds.view(b_sz, n_attrs*n_anchors, grid_size*grid_size) # flatten 2-d grid
    preds = preds.transpose(1, 2).contiguous() # new shape (b_sz, row_order_pixels, n_attrs*n_anchors)
    preds = preds.view(b_sz, grid_size*grid_size*n_anchors, n_attrs) # new shape (b_sz, row_order_pixels * n_anchors, n_attrs)

    # Scale down anchor positions to feature map resolution
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Compute deltas for x_center, y_center
    # Compute object confidence (i.e. is there a relevant object there?)
    preds[:, :, 0] = torch.sigmoid(preds[:, :, 0])
    preds[:, :, 1] = torch.sigmoid(preds[:, :, 1])
    preds[:, :, 4] = torch.sigmoid(preds[:, :, 4])

    # Add offsets to the x,y center positions
    grid = np.arange(grid_size)
    h_axis, w_axis = np.meshgrid(grid, grid)
    x_axis, y_axis = torch.FloatTensor(h_axis).view(-1, 1), torch.FloatTensor(w_axis).view(-1, 1)
    if device != None:
        x_axis, y_axis = x_axis.to(device), y_axis.to(device)
    x_y_grid = torch.cat((x_axis, y_axis), dim=1).repeat(1, n_anchors).view(-1, 2).unsqueeze(dim=0) # merge x,y offsets, repeat for each anchor; (1, grid_size*grid_size*n_anchors, 2)
    preds[:, :, :2] += x_y_grid

    # Apply anchors to bounding box dimensions
    anchors = torch.FloatTensor(anchors)
    if device != None:
        anchors = anchors.to(device)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(dim=0)
    preds[:, :, 2:4] = torch.exp(preds[:, :, 2:4])*anchors

    # Sigmoid on class scores
    preds[:, :, -n_classes:] = torch.sigmoid(preds[:, :, -n_classes:])

    # Transform center coordinates and anchor box dimensions back to input dims
    preds[:, :, :4] *= stride

    return preds

def get_detections(preds: torch.Tensor, confidence: float, nms_conf: float = 0.4) -> Optional[torch.Tensor]:
    # Initialize output
    output = None

    # Mask out bounding boxes below confidence threshold
    conf_mask = (preds[:, :, 4] > confidence).float().unsqueeze(2)
    preds = preds*conf_mask

    # Convert center coordinates and bbox height/width to boundary coordinates
    box_bounds = preds[:, :, :4].clone().detach()
    box_bounds[:, :, 0] = (preds[:, :, 0] - preds[:, :, 2]/2) # left
    box_bounds[:, :, 1] = (preds[:, :, 1] - preds[:, :, 3]/2) # bottom
    box_bounds[:, :, 0] = (preds[:, :, 0] + preds[:, :, 2]/2) # right
    box_bounds[:, :, 1] = (preds[:, :, 1] + preds[:, :, 2]/2) # top
    preds[:, :, :4] = box_bounds

    b_sz = preds.shape[0]
    for idx in range(b_sz):
        img_preds = preds[idx]

        # Detect all objects in image
        max_conf, max_conf_idxs = torch.max(img_preds[:, 5:], dim=1)
        max_conf, max_conf_idxs = max_conf.float().unsqueeze(dim=1), max_conf_idxs.float().unsqueeze(dim=1)
        seq = (img_preds[:, :5], max_conf, max_conf_idxs) # 7 features per box -> bbox bounds (4x), object confidence, confidence score, and idxs
        img_preds = torch.cat(seq, dim=1)

        # Remove bounding boxes below confidence threshold
        nonzero_idxs = torch.nonzero(img_preds[:, 4]).squeeze()
        if nonzero_idxs.numel() > 0:
            img_preds = img_preds[nonzero_idxs]

        # Get the object predictions
        img_classes = get_unique_classes(img_preds[:, -1])

        # Perform class-wise non-maximum suppression (NMS), IoU
        for cls in img_classes:
            # Get detections for class
            cls_mask = img_preds*((img_preds[:, -1] == cls).float().unsqueeze(dim=1))
            cls_mask = torch.nonzero(cls_mask[:, -1]).squeeze()
            img_preds_cls = img_preds[cls_mask]

            # Sort bounding boxes for class from highest to lowest
            conf_sort_idxs = img_preds_cls[:, 4].sort(descending=True)[1]
            img_preds_cls = img_preds_cls[conf_sort_idxs]

            # NMS -> drops identical detections of same object class via IoU (i.e. if two boxes detect same tree, remove the one w/ lower object confidence)
            j = 0
            while (img_preds_cls.shape[0] > 1) and (j < img_preds_cls.shape[0] - 1):
                # Get current bbox, remaining bboxs
                curr_bbox = img_preds_cls[j].unsqueeze(dim=0)
                rem_bboxs = img_preds_cls[j+1:]

                # Compute IoUs
                ious = get_bbox_iou(curr_bbox, rem_bboxs)

                # Remove bboxes w/ IoU < nms_conf
                low_nms_mask = (ious < nms_conf).float().unsqueeze(1)
                img_preds_cls[j+1:] *= low_nms_mask
                high_nms_idxs = torch.nonzero(img_preds_cls[:, 4]).squeeze()
                img_preds_cls = img_preds_cls[high_nms_idxs]

                j += 1

            # Write detections of class to output
            batch_idxs = img_preds_cls.new_full((img_preds_cls.shape[0], 1), fill_value=idx) # get each unique bbox with low NMS
            seq = torch.cat((batch_idxs, img_preds_cls), dim=1)

            if output is None:
                output = seq
            else:
                output = torch.cat((output, seq), dim=0)

    return output

def get_bbox_iou(curr_bbox: torch.Tensor, rem_bboxs: torch.Tensor) -> torch.Tensor:
    # Get left, bottom, right, top bounds
    cb_x1, cb_y1, cb_x2, cb_y2 = curr_bbox[:, 0], curr_bbox[:, 1], curr_bbox[:, 2], curr_bbox[:, 3]
    rb_x1, rb_y1, rb_x2, rb_y2 = rem_bboxs[:, 0], rem_bboxs[: ,1], rem_bboxs[:, 2], rem_bboxs[:, 3]

    # Get intersection region between curr_bbox and all rem_bboxs
    inter_x1 = torch.max(cb_x1, rb_x1) # inter left
    inter_y1 = torch.max(cb_y1, rb_y1) # inter bottom
    inter_x2 = torch.min(cb_x2, rb_x2) # inter right
    inter_y2 = torch.min(cb_y2, rb_y2) # inter top
    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0)*torch.clamp(inter_y2 - inter_y1 + 1, min=0)

    # Get union regions
    cb_area = (cb_x2 - cb_x1 + 1)*(cb_y2 - cb_y1 + 1)
    rb_area = (rb_x2 - rb_x1 + 1)*(rb_y2 - rb_y1 + 1)

    # IoU
    return inter_area / (cb_area + rb_area - inter_area)

def get_unique_classes(class_idxs: torch.Tensor) -> torch.Tensor:
    # Get unique class idxs
    unique_classes = torch.from_numpy(np.unique(class_idxs.cpu().detach().numpy()))
    
    return unique_classes.clone().detach()

def resize_image(img: np.ndarray, res: int) -> np.ndarray:
    """
    Resize OpenCV image to preserve original aspect ratio, overlay on greyscale canvas of desired resolution.
    """
    # Get new width/height, keep aspect ratio constant
    w, h = img.shape[1], img.shape[2]
    scale = min(res/w, res/h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create greyscale canvas (desired res)
    canvas = np.full((res, res, 3), 128)

    # Place new_img in center
    canvas[(h - new_h)//2:(h - new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = new_img

    return canvas

def prep_image(img: np.ndarray, res: int) -> np.ndarray:
    """
    Takes a resized image and prepares input to YOLOv3.
    """
    # Resize image
    new_img = cv2.resize(img, (res, res))

    # Input dims: (H x W x BGR) -> (RGB x H x W)
    new_img = new_img[:, :, ::-1].transpose((2, 0, 1)).copy()

    # Convert to tensor
    new_img = torch.from_numpy(new_img).float().unsqueeze(dim=0)/255.0

    return new_img