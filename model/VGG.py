from torch import nn
from typing import Tuple

class Conv2DBlock():
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding):

        self.block_list=[nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding),
        nn.ReLU()]

class VGGBlockA():
    def __init__(self,number_of_convs,input_channels,output_channels,kernel,stride,padding,maxpool_kernel_size,maxpool_stride):

        self.block_list=[]
        self.input_channels=input_channels
        self.output_channels=output_channels
        for _ in range(number_of_convs):
            self.block_list.extend(Conv2DBlock(self.input_channels,self.output_channels,kernel,stride,padding).block_list)
            self.input_channels=self.output_channels
        self.block_list.append(nn.MaxPool2d(maxpool_kernel_size,maxpool_stride))

class FCLayerWithDropOut():
    def __init__(self,input_features,output_features,DropOutRate):
        self.layer_list=[
            nn.Linear(input_features,output_features),
            nn.ReLU(),
            nn.Dropout(DropOutRate)
        ]

class VGGModel_11(nn.Module):
    def __init__(self,in_features:int,image_resolution:int,out_features:int,Architechture=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        super().__init__()
        self.block=[]
        self.in_featues=in_features
        for (num_convs,num_channels) in Architechture:
            self.out_features=num_channels
            self.block.append(nn.Sequential(*VGGBlockA(num_convs,self.in_featues,self.out_features,3,1,1,2,2).block_list))
            self.in_featues=num_channels
        print(self.block)

        self.block=nn.Sequential(*self.block)
        self.flatten=nn.Flatten()
        self.FC1=nn.Sequential(*FCLayerWithDropOut(2048,4096,0.5).layer_list)
        self.FC2=nn.Sequential(*FCLayerWithDropOut(4096,4096,0.5).layer_list)
        self.FC3=nn.Linear(4096,10)

    def forward(self,x):
        x=self.block(x)
        x=self.flatten(x)
        x=self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        ##x=self.softmax(x)
        return x



class CNNModel_E(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x
