from pathlib import Path
from data_setup import createDataLoaders
from torchvision import transforms
import torch
from model_builder import CNNModel_A
from engine import train

def main2():
    model = CNNModel_A(in_features=3,hiddent_units=64,out_features=10)
    print(model)

def main():
    train_dir=str(Path("data")/"train")
    test_dir = str(Path("data") / "test")
    transform=transforms.Compose(transforms.ToTensor())
    batch_size=32
    num_workers=torch.cpu.device_count()
    train_data,test_data,class_names=createDataLoaders(train_dir,test_dir,transform,batch_size,num_workers)
    if torch.cuda:
        device=torch.cuda
    else:
        device=torch.device("mps")

    model=CNNModel_A(in_features=3,hiddent_units=64,out_features=10)
    learningRate=0.01
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate)
    loss_fn=torch.nn.L1Loss()
    train(5,model,optimizer,loss_fn,train_data,test_data,device)






if __name__=='__main__':
    main()