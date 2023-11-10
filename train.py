from pathlib import Path
from data_setup import createDataLoaders
from torchvision import transforms
import torch
from model.VGG import VGGModel_11,TinyVGG
from engine import train
from torch.utils.tensorboard import SummaryWriter

def main2():
    model = VGGModel_11(in_features=3,image_resolution=64,out_features=10)
    print(model)

def main():
    train_dir=Path("data")/"train"
    test_dir = Path("data") / "test"
    transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()])

    batch_size=256
    in_shape=3
    image_resolution=64
    out_shape=20
    epoches=80
    num_workers=torch.cuda.device_count()

    train_data,test_data,class_names=createDataLoaders(train_dir,test_dir,transform,batch_size,num_workers)
    if torch.cuda.is_available():
        device=torch.device("cuda:3")
    else:
        device=torch.device("mps")

   # device=torch.device("cpu")


    model=VGGModel_11(in_features=in_shape,image_resolution=image_resolution,out_features=out_shape).to(device)
    #model=TinyVGG(input_shape=3,output_shape=20,hidden_units=10).to(device)
    learningRate=0.01
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate)
    loss_fn=torch.nn.CrossEntropyLoss()
    writer=SummaryWriter()
    result=train(epoches,model,loss_fn,optimizer,train_data,test_data,device,writer)







if __name__=='__main__':
    main()