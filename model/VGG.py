from torch import nn
from typing import Tuple

class Conv2DBlock():
    def __init__(self,input_channels,output_channels,kernel_size,stride,padding):

        self.block_list=[nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding),
                         nn.BatchNorm2d(output_channels)
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
    def __init__(self,in_features:int,image_resolution:int,out_features:int,Architecture=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        super().__init__()
        self.block=[]
        self.in_featues=in_features
        for (num_convs,num_channels) in Architecture:
            self.out_features=num_channels
            self.block.append(nn.Sequential(*VGGBlockA(num_convs,self.in_featues,self.out_features,3,1,1,2,2).block_list))
            self.in_featues=num_channels


        self.block=nn.Sequential(*self.block)
        self.flatten=nn.Flatten()
        fc1_in=self.out_features * (image_resolution//(2**len(Architecture))) * (image_resolution//(2**len(Architecture)))
        self.FC1=nn.Sequential(*FCLayerWithDropOut(fc1_in,4096,0.5).layer_list)
        self.FC2=nn.Sequential(*FCLayerWithDropOut(4096,4096,0.5).layer_list)
        self.FC3=nn.Linear(4096,out_features)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.block(x)
        x=self.flatten(x)
        #print(x.shape)
        x=self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x=self.softmax(x)
        return x



class TinyVGG(nn.Module):
    def __init__(self,input_shape:int,output_shape:int,hidden_units:int):
        super().__init__()

        self.conv_block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*(hidden_units//(2**2))*(hidden_units//(2**2)),out_features=output_shape)
        )

    def forward(self,x):
        x=self.conv_block1(x)
        x=self.conv_block2(x)
        x=self.classifier(x)

        return x



class CNNModel_E(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x
