from torch import nn

class CNNModel_A(nn.Module):
    def __init__(self,in_features:int,hiddent_units:int,out_features:int):
        super().__init__()
        self.CNNBlock_1=nn.Sequential(nn.Conv2d(in_features,hiddent_units,3,1,1),
                                      nn.ReLU(),
                                   nn.MaxPool2d(3,1,1)
                                   )
        self.CNNBlock_2=nn.Sequential(nn.Conv2d(hiddent_units,hiddent_units*2,3,1,1),
                                      nn.ReLU(),
                                   nn.MaxPool2d(3,1,1))

        self.CNNBlock_3 = nn.Sequential(nn.Conv2d(hiddent_units*2, hiddent_units*2, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(hiddent_units * 2, hiddent_units * 4, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 1, 1))

        self.CNNBlock_4 = nn.Sequential(nn.Conv2d(hiddent_units*4, hiddent_units*8, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(hiddent_units * 8, hiddent_units * 8, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 1, 1))

        self.CNNBlock_5 = nn.Sequential(nn.Conv2d(hiddent_units*8, hiddent_units*8, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(hiddent_units*8, hiddent_units*8, 3, 1, 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 1, 1))

        self.flatten=nn.Flatten(1,1)
        self.FC1=nn.Sequential(nn.Linear(hiddent_units*32,hiddent_units*64),
                                   nn.ReLU(), nn.Dropout(0.5))
        self.FC2=nn.Sequential(nn.Linear(in_features=hiddent_units*64,out_features=hiddent_units*64),
                                   nn.ReLU(), nn.Dropout(0.5))
        self.FC3 = nn.Linear(in_features=hiddent_units*64,out_features=out_features)
        ##self.softmax=nn.Softmax2d()
    def forward(self,x):
        x=self.CNNBlock_1(x)
        x=self.CNNBlock_2(x)
        x = self.CNNBlock_3(x)
        x = self.CNNBlock_4(x)
        x = self.CNNBlock_5(x)
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
