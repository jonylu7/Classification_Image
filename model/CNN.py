from torch import nn
from VGG import Conv2DBlock
class EasyConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
