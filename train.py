from pathlib import Path
from data_setup import createDataLoaders
from torchvision import transforms
import torch
from model.VGG import VGGModel_11
from engine import train

def main2():
    model = VGGModel_11(in_features=3,image_resolution=64,out_features=10)
    print(model)

def main():
    train_dir=Path("data")/"train"
    test_dir = Path("data") / "test"
    transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()])
    batch_size=32
    num_workers=torch.cuda.device_count()
    train_data,test_data,class_names=createDataLoaders(train_dir,test_dir,transform,batch_size,num_workers)
    if torch.cuda.is_available():
        device=torch.device("cuda:3")
    else:
        device=torch.device("mps")

    model=VGGModel_11(in_features=3,image_resolution=64,out_features=10).to(device)
    learningRate=0.01
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate)
    loss_fn=torch.nn.CrossEntropyLoss()

    train(5,model,loss_fn,optimizer,train_data,test_data,device)






if __name__=='__main__':
    main()