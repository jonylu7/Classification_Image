from pathlib import Path
from data_setup import createDataLoaders
from torchvision import transforms,datasets
import torch
from model.VGG import VGGModel_11,TinyVGG
from engine import train
from torch.utils.tensorboard import SummaryWriter

def main2():
    model = VGGModel_11(in_features=3,image_resolution=64,out_features=10)
    print(model)

def customizeDataSets():
    train_dir = Path("data") / "train"
    test_dir = Path("data") / "test"
    batch_size = 256
    num_workers = torch.cuda.device_count()
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
    train_data, test_data, class_names = createDataLoaders(train_dir, test_dir, transform, batch_size, num_workers)
    return  train_data, test_data, class_names

def MNISTDatasets():
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),transforms.Normalize(0.5,0.5)])
    train_data=datasets.MNIST(root="data/MNIST",train=True,download=False,transform=transform)
    test_data=datasets.MNIST(root="../data/MNIST",train=False,download=False,transform=transform)
    class_names=train_data.classes
    return train_data,test_data,class_names

def main(train_data,test_data,in_shape,image_resolution,out_shape):

    weight_decay=0.0005
    epoches=10



    if torch.cuda.is_available():
        device=torch.device("cuda:0")
    else:
        device=torch.device("mps")

   # device=torch.device("cpu")

    model=VGGModel_11(in_features=in_shape,image_resolution=image_resolution,out_features=out_shape).to(device)
    #model=TinyVGG(input_shape=3,output_shape=20,hidden_units=10).to(device)
    learningRate=0.0005
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate,weight_decay=weight_decay,momentum=0.9)
    loss_fn=torch.nn.CrossEntropyLoss()
    writer=SummaryWriter()
    result=train(epoches,model,loss_fn,optimizer,train_data,test_data,device,writer)







if __name__=='__main__':
    train_data, test_data, class_names = customizeDataSets()
    #train_data, test_data, =MNISTDatasets()
    main(train_data, test_data)