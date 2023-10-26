from pathlib import Path
from data_setup import createDataLoaders
from torchvision import transforms
import torch
from model_builder import CNNModel_1
from engine import train

def main():
    train_dir=str(Path("data")/"train")
    test_dir = str(Path("data") / "test")
    transform=transforms.Compose(transforms.ToTensor())
    batch_size=32
    num_workers=torch.cpu.device_count()
    train_data,test_data,class_names=createDataLoaders(train_dir,test_dir,transform,batch_size,num_workers)


    model=CNNModel_1()
    learningRate=0.01
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate)
    loss_fn=torch.nn.L1Loss()
    train(5,optimizer,loss_fn)






if __name__=='__main__':
    main()