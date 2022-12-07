from typing import Optional
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

from .utils import predict_transform, get_detections
from .modules import EmptyModule, DetectionModule

class Darknet(nn.Module):
    def __init__(self, cfg_file: Path, device: Optional[torch.device] = None) -> None:
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_blocks(self.blocks)
        self.device = device
        if self.device != None:
            self.module_list.to(self.device)

    def modify_net_info(self, **kwargs):
        """
        Manually override default net_info parameters from config.
        """
        for k,v in kwargs.items():
            self.net_info[k] = v

    def load_weights(self, weights_file: Path):
        """
        Read in pre-trained model weight file.
        """
        with open(weights_file, "rb") as f:
            # Store header info (first 5 lines)
            header = np.fromfile(f, dtype = np.int32, count=5)
            self.header = torch.tensor(header)
            self.seen = self.header[3] # subversion number

            # Load weights
            weights = np.fromfile(f, dtype=np.float32)
            wt_pointer = 0
            for idx in range(len(self.module_list)):
                # Get module type
                module_type = self.blocks[idx+1]["type"]

                if module_type == "convolutional":
                    # Get module and conv layer
                    module = self.module_list[idx]
                    conv = module[0]

                    # Load batch norm params (if exists)
                    batch_norm = int(self.blocks[idx+1].get("batch_normalize", 0))
                    if batch_norm:
                        # Get all batch norm parameters
                        bn = module[1]
                        bn_params = [bn.bias.data, bn.weight.data, bn.running_mean, bn.running_var]

                        # Get number of weights in batch norm layer
                        n_wts = bn.bias.numel()

                        # Load batch norm biases, weights, running means, running variances
                        for i, bnp in enumerate(bn_params):
                            # Get pretrained params, increment pointer
                            pretrained_params = torch.from_numpy(weights[wt_pointer: wt_pointer+n_wts])
                            wt_pointer += n_wts

                            # Cast parameters into model dimensions
                            pretrained_params =  pretrained_params.view_as(bnp)

                            # Copy parameters over
                            bnp.copy_(pretrained_params)
                    # Load conv biases
                    else:
                        n_wts = conv.bias.data.numel()
                        conv_bias_pretrained = torch.from_numpy(weights[wt_pointer: wt_pointer+n_wts])
                        wt_pointer += n_wts

                        conv_bias_pretrained = conv_bias_pretrained.view_as(conv.bias.data)
                        conv.bias.data.copy_(conv_bias_pretrained)
                    
                    # Load conv weights
                    n_wts = conv.weight.numel()
                    conv_weight_pretrained = torch.from_numpy(weights[wt_pointer: wt_pointer+n_wts])
                    wt_pointer += n_wts

                    conv_weight_pretrained = conv_weight_pretrained.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weight_pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get block_dicts (element 0 is net_info)
        block_dicts = self.blocks[1:]
        
        # Initialize output cache
        outputs = {}

        # Forward pass
        write = 0
        for idx, module in enumerate(block_dicts):
            # Get module type
            module_type = (module["type"])

            # Each module forward pass
            if (module_type == "convolutional") or (module_type == "upsample"):
                x = self.module_list[idx](x.to(self.device))
            elif module_type == "route":
                # Get layers to route skip connections
                layers = module["layers"]
                layers = [int(a) for a in layers.split(", ")]

                # Get single output if one layer specified
                if len(layers) == 1:
                    x = outputs[idx + layers[0]]

                # Concatenate two outputs
                else:
                    out_1 = outputs[idx + layers[0]]
                    out_2 = outputs[layers[1]]
                    x = torch.cat((out_1, out_2), dim=1)
            elif module_type == "shortcut":
                # Get layer_idx
                layer_idx = int(module["from"])

                # Get prior layer output
                prior_x = outputs[idx + layer_idx]

                # Residual connection w/ prior layer output
                x = outputs[idx- 1] + prior_x
            elif module_type == "yolo":
                # Get anchor boxes, input_dims, num_classes
                input_dims = int(self.net_info["height"])
                anchors = self.module_list[idx][0].anchors
                n_classes = int(module["classes"])

                # Get YOLO predictions
                x = predict_transform(x, input_dims, anchors, n_classes, self.device)

                # Concatenate predictions with all resolutions
                if not write:
                    detections = x
                    write =1
                else:
                    detections = torch.cat((detections, x), dim=1)

            # Append x to outputs
            outputs[idx] = x

        return detections

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

if __name__ == "__main__":
    # Initialize darknet model
    cfg_file = Path("../cfg/yolov3.cfg")
    wts_file = Path("../data/yolov3.weights")

    # Get available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    darknet = Darknet(cfg_file)

    # Load model weights
    print("loading pretrained weights...")
    darknet.load_weights(wts_file)

    # Test forward pass
    print("testing forward pass...")
    def get_test_img():
        # Read and resize image to square
        img = cv2.imread("../data/dog-cycle-car.png")
        img = cv2.resize(img, (416, 416))

        # Convert BGR -> RGB, move channel dim to front (C X H X W), add a batch dimension
        img = img[:, :, ::-1].transpose((2, 0, 1))
        img = img[np.newaxis, :, :, :]

        # Normalize (0 to 1), convert to torch
        img = torch.tensor(img/255.0, dtype=torch.float32)
        return img
    
    x = get_test_img()

    # Modify net_info
    darknet.modify_net_info(height=416, width=416)

    preds = darknet(x)
    print("Output shape (batch size, # bbox, # bbox features) - ", preds.shape)

    # Get detections
    detections = get_detections(preds, 0.5)
    print("Number of unique detections in image: ", detections.shape[0])