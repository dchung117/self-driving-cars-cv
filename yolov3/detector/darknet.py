from typing import Optional
from pathlib import Path

import numpy as np
import cv2
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

    detections = darknet(x)
    print("Output shape (batch size, # bbox, # bbox features): ", detections.shape)