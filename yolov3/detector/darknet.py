from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils import parse_cfg, create_blocks, predict_transform

class Darknet(nn.Module):
    def __init__(self, cfg_file: Path, device: Optional[torch.device] = None) -> None:
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_blocks(self.blocks)
        self.device = device
        if self.device != None:
            self.module_list.to(self.device)

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
                x = self.module_list[idx](x.to(self.device)).cpu()
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
                print(x.shape)
                x = predict_transform(x, input_dims, anchors, n_classes, self.device)
                print(x.shape)
                # Concatenate predictions with all resolutions
                if not write:
                    detections = x
                    write =1
                else:
                    detections = torch.cat((detections, x), dim=1)

            # Append x to outputs
            outputs[idx] = x

        return detections

if __name__ == "__main__":
    # Initialize darknet model
    cfg_file = Path("../cfg/yolov3.cfg")
    
    # Get available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    darknet = Darknet(cfg_file)

    # Test forward pass
    b_sz = 64
    n_channels = 3
    height = 608
    width = 608
    x = torch.rand(b_sz, n_channels, height, width)
    detections = darknet(x)
    print(detections.shape)