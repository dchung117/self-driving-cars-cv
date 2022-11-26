from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# place holder for skip connections
class EmptyModule(nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()

class DetectionModule(nn.Module):
    def __init__(self, anchors: list[tuple]) -> None:
        super(DetectionModule, self).__init__()
        self.anchors = anchors

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
    Create skip connection with current output with block that is 3 connections upstream.
    """
    shortcut = EmptyModule()
    block.add_module(f"shortcut_{idx}", shortcut)

    return block

def _create_route_block(block: nn.Sequential, block_dict: dict, output_filters: list, idx: int) -> tuple[nn.Sequential, int]:
    """
    Create skip connections for layer(s) that are more than 4 blocks upstream.
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

if __name__ == "__main__":
    # Parse config file
    cfg_file = Path("../cfg/yolov3.cfg")
    blocks = parse_cfg(cfg_file)

    # Create modules blcocks
    net_info, module_list = create_blocks(blocks)
    for m in module_list:
        print(m)