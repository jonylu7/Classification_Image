from torch import nn

class ResNet_1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x