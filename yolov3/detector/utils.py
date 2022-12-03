from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from modules import EmptyModule, DetectionModule

def parse_cfg(cfg_file: Path) -> list[dict]:
    """
    Reads in YOLOv3 configuration file. 

    Returns list of dictionaries. Each dictionary represents a block in the network.
    """
    # Read config file
    with open(cfg_file, "r") as f:
        lines = f.read().split("\n")
        
        # Remove empty lines, comments, extra whitespace
        lines = [l.rstrip().lstrip() for l in lines if len(l) > 0]
        lines = [l.rstrip().lstrip() for l in lines if l[0] != "#"]

    # Create blocks
    blocks = []
    block = {}
    for l in lines:
        # Check if new block
        if l[0] == "[":
            if len(block) != 0:
                # Store current block
                blocks.append(block)

                # Create new block
                block = {}
                block["type"] = l[1:-1].rstrip()

        else: # Append value to block
            key, value = l.split("=")
            key = key.rstrip()
            value = value.lstrip()
            block[key] = value
    blocks.append(block)

    return blocks

def create_blocks(block_list: list[dict]) -> tuple[dict, nn.ModuleList]:
    # First element is network info
    net_info = block_list[0]

    # Initialize module list
    module_list = nn.ModuleList()
    prev_filters = 3 # input is RGB image
    output_filters = []

    # Create modules from block_list
    for idx, block_info in enumerate(block_list[1:]):
        # Initialize module
        module = nn.Sequential()

        # Convolutional block
        if block_info["type"] == "convolutional":
            module, filters = _create_conv_block(module, block_info, idx, prev_filters)
        # Upsampling block
        elif block_info["type"] == "upsample":
            module = _create_upsample_block(module, block_info, idx)
        # Routing layer
        elif block_info["type"] == "route":
            module, filters = _create_route_block(module, block_info, output_filters, idx)
        # Shortcut layer (always skip connection from 3rd prior conv block)
        elif block_info["type"] == "shortcut":
            module = _create_shortcut_block(module, idx)
        # YOLO detection layer
        elif block_info["type"] == "yolo":
            module = _create_yolo_block(module, block_info, idx)

        # Append module to module_list
        module_list.append(module)

        # Update filters
        prev_filters = filters

        # Append out filters
        output_filters.append(filters)

    return net_info, module_list

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

def _create_conv_block(block: nn.Sequential, block_dict: dict, idx: int, in_filters: int) -> tuple[nn.Sequential, int]:
    """
    Create 2d convolutional block from block_dict. Uses kernel size, stride length, padding, and/or batch normalization, activation function.
    """
    # Parse batch_normalize, filters, kernel size, stride, padding into ints
    out_filters = int(block_dict["filters"])
    kernel_size = int(block_dict["size"])
    stride = int(block_dict["stride"])

    bn = None
    bias = True
    if "batch_normalize" in block_dict:
        batch_normalize = int(block_dict["batch_normalize"])
        if batch_normalize:
            bn = nn.BatchNorm2d(out_filters)
            bias = False


    # same padding
    pad = int(block_dict["pad"])
    if pad:
        pad = (kernel_size - 1) // 2
    else:
        pad = 0

    # Create convolutional layer
    conv2d = nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=pad, bias=bias)
    block.add_module(f"conv_{idx}", conv2d)

    # Create batch normalization layer (if needed)
    if bn != None:
        block.add_module(f"batch_norm_{idx}", bn)

    # Create activation layer
    if block_dict["activation"] == "leaky":
        activation = nn.LeakyReLU(0.1, inplace=True)
        block.add_module(f"leaky_{idx}", activation)
    
    return block, out_filters

def _create_upsample_block(block: nn.Sequential, block_dict: dict, idx: int) -> nn.Sequential:
    """
    Create upsampling block via bilinear interpolation. The stride key-value pair specifies the scale factor.
    """
    # Get the scale factor, bilinear interpolation
    scale_factor= int(block_dict["stride"])
    upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
    block.add_module(f"upsample_{idx}", upsample)

    return block

def _create_shortcut_block(block: nn.Sequential, idx:int) -> nn.Sequential:
    """
    Create skip connection (residual addition) with current output with block that is 3 connections upstream.
    """
    shortcut = EmptyModule()
    block.add_module(f"shortcut_{idx}", shortcut)

    return block

def _create_route_block(block: nn.Sequential, block_dict: dict, output_filters: list, idx: int) -> tuple[nn.Sequential, int]:
    """
    Create skip connections (concatenation) for layer(s) that are more than 4 blocks upstream.
    """
    # Parse out routing layers
    layers = [int(x) for x in block_dict["layers"].split(",")]

    # Get start point of route
    start_idx = layers[0]

    # Get end point of route (if exists)
    if len(layers) == 2:
        end_idx = layers[1]
    else:
        end_idx = 0

    if end_idx > 0:
        end_idx = end_idx - idx

    # Add routing module
    route = EmptyModule()
    block.add_module(f"route_{idx}", route)

    # Update number of filters by from previous layers
    filters = output_filters[idx + start_idx]
    if end_idx < 0:
        filters = filters + output_filters[idx + end_idx]

    return block, filters

def _create_yolo_block(block: nn.Sequential, block_dict: dict, idx: int) -> nn.Sequential:
    # Get mask idxs
    mask_idxs = [int(x) for x in block_dict["mask"].split(",")]

    # Get anchor box coord list and mask_idxs
    anchors = [x.split(",") for x in block_dict["anchors"].split(", ")]
    anchors = [tuple([int(x) for x in a]) for a in anchors]
    anchors = [anchors[i] for i in mask_idxs]

    # Create detection module
    yolo = DetectionModule(anchors)
    block.add_module(f"detection_{idx}", yolo)

    return block
