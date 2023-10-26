from pathlib import Path
from data_setup import createDataLoaders
from torchvision import transforms
import torch

def main():
    train_dir=str(Path("data")/"train")
    test_dir = str(Path("data") / "test")
    transform=transforms.Compose(transforms.ToTensor())
    batch_size=32
    num_workers=torch.cpu.device_count()
    train_data,test_data,class_names=createDataLoaders(train_dir,test_dir,transform,batch_size,num_workers)






if __name__=='__main__':
    main()