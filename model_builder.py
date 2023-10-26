from torch import nn

class CNNModel_1(nn.Module):
    def __init__(self,in_features:int,hiddent_units:int,out_features:int):
        super().__init__()

    def forward(self,x):
        return x


class ResNet_1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class VGG(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x