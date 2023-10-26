from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def createDataLoaders(train_dir:str,test_dir:str,transform:transforms.Compose,batch_size:int,num_workers):

    train_data_folder=datasets.ImageFolder(train_dir,transform=transform)
    test_data_folder=datasets.ImageFolder(test_dir,transform=transform)

    train_data=DataLoader(train_data_folder,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    test_data=DataLoader(test_data_folder,batch_size=batch_size,shuffle=True,num_workers=num_workers)


    class_names=train_data_folder.classes

    return train_data,test_data,class_names

