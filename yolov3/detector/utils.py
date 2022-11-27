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
