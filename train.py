from pathlib import Path

import torchvision.models

from data_setup import createDataLoaders
from torchvision import transforms,datasets
import torch
from model.VGG import VGGModel_11,TinyVGG
from engine import train
from torch.utils.tensorboard import SummaryWriter
from utils import save_model

def main2():
    model = VGGModel_11(in_features=3,image_resolution=64,out_features=10)
    print(model)

def customizeDataSets():
    train_dir = Path("data") / "train"
    test_dir = Path("data") / "test"
    batch_size =256
    num_workers = torch.cuda.device_count()
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    train_data, test_data, class_names = createDataLoaders(train_dir, test_dir, transform, batch_size, num_workers)
    return  train_data, test_data, class_names

def MNISTDatasets(downloaded,batch_size,image_resolution):
    transform = transforms.Compose([transforms.Resize(image_resolution), transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
    train_data_sets=datasets.MNIST(root="data/MNIST/train",train=True,download=downloaded,transform=transform)
    test_data_sets=datasets.MNIST(root="data/MNIST/test",train=False,download=downloaded,transform=transform)
    train_data=torch.utils.data.DataLoader(dataset=train_data_sets,batch_size=batch_size,shuffle=True)
    test_data=torch.utils.data.DataLoader(dataset=test_data_sets,batch_size=batch_size,shuffle=True)
    class_names=train_data_sets.classes
    return train_data,test_data,class_names




def CIFARDatasets(downloaded,batch_size,image_resolution):
    transform = transforms.Compose([transforms.Resize(image_resolution), transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
    train_data_sets=datasets.CIFAR10(root="data/CIFAR/train",train=True,download=downloaded,transform=transform)
    test_data_sets=datasets.CIFAR10(root="data/CIFAR/test",train=False,download=downloaded,transform=transform)
    train_data=torch.utils.data.DataLoader(dataset=train_data_sets,batch_size=batch_size,shuffle=True)
    test_data=torch.utils.data.DataLoader(dataset=test_data_sets,batch_size=batch_size,shuffle=True)
    class_names=train_data_sets.classes
    return train_data,test_data,class_names



def main(train_data,test_data,in_shape,image_resolution,out_shape):

    weight_decay=0.00005
    epoches=74
    momentum=0.9


    if torch.cuda.is_available():
        device=torch.device("cuda:1")
    else:
        device=torch.device("mps")

   # device=torch.device("cpu")

    model=VGGModel_11(in_features=in_shape,image_resolution=image_resolution,out_features=out_shape).to(device)

    #model.load(torchvision.models.VGG11_Weights)

    #model.apply(torchvision.models.VGG11_Weights)
    #model=TinyVGG(input_shape=in_shape,output_shape=out_shape,hidden_units=image_resolution).to(device)
    learningRate=0.002
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate,weight_decay=weight_decay,momentum=momentum)
    loss_fn=torch.nn.CrossEntropyLoss()
    writer=SummaryWriter()

    result=train(epoches,model,loss_fn,optimizer,train_data,test_data,device,writer)
    save_model(model,"Save_model","VGG_11_3.pth")










if __name__=='__main__':
    #train_data, test_data, class_names = customizeDataSets()
    train_data, test_data,class_names =MNISTDatasets(downloaded=False,batch_size=256,image_resolution=28)
    #train_data,test_data,class_names=ImageNetDatasets(downloaded=True,batch_size=256,image_resolution=224)
    main(train_data, test_data,1,28,len(class_names))