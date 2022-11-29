import torch.nn as nn

# place holder for skip connections
class EmptyModule(nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()

class DetectionModule(nn.Module):
    def __init__(self, anchors: list[tuple]) -> None:
        super(DetectionModule, self).__init__()
        self.anchors = anchors