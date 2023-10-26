import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils import unpickle,readClasses

def turnDataToImageFile(path_father:Path,path_son:str,path_save:Path,classnames:list,superclass:bool = True):
    if(path_save.is_dir()==False):
        path_save.mkdir(parents=True)
    data_path=path_father/path_son
    data=unpickle(data_path)
    datalen=len(data[b'data'])

    imageShape = (32, 32)
    for i in range(datalen):
        image_data = np.zeros((32, 32, 3), dtype=np.uint8)
        red = np.array(data[b'data'][i][:1024])
        green = np.array(data[b'data'][i][1024:2048])
        blue = np.array(data[b'data'][i][2048::])
        image_data[:, :, 0] = red.reshape(imageShape)
        image_data[:, :, 1] = green.reshape(imageShape)
        image_data[:, :, 2] = blue.reshape(imageShape)
        if superclass:
            Image.fromarray(image_data).save(path_save/classnames[data[b'coarse_labels'][i]]/str(data[b'filenames'][i])[2:-1])
        else:
            Image.fromarray(image_data).save(path_save / classnames[data[b'fine_labels'][i]] / str(data[b'filenames'][i])[2:-1])

def mkdir_classes(path_save:Path,classes:list):
    for c in classes:
        p=path_save/c
        if (p.is_dir() == False):
            p.mkdir(parents=True)


def unpackFiles(train_or_test:str,classes:list,father_path:Path):

    saving_path = Path("data") / train_or_test
    mkdir_classes(saving_path, classes)

    mkdir_classes(saving_path, classes)
    turnDataToImageFile(father_path/"cifar-100-python", train_or_test, Path("data") / train_or_test, classes, True)
    turnDataToImageFile(father_path/"cifar-100-python", train_or_test, Path("data") / train_or_test, classes, True)


if __name__=='__main__':
    father_path = Path("data") / "CIFAR_100/"
    file = father_path / Path("cifar-100-python") / "meta"
    classes = readClasses(file, True)
    unpackFiles("train",classes,father_path)
    unpackFiles("test", classes, father_path)

def testFunction():
    #just to test reading data
    test_data_path=Path("cifar-100-python")/"test"
    test_data=unpickle(test_data_path)

    print(list(test_data.keys()))

    for elements in test_data.keys():
        print(test_data[elements][0])


    imageShape=(32,32)
    fig=plt.figure(figsize=(10,5))

    meta_data_path=Path("cifar-100-python")/"meta"
    meta_data=unpickle(meta_data_path)

    for i in range(32):
        image_data = np.zeros((32, 32, 3), dtype=np.uint8)
        red=np.array(test_data[b'data'][i][:1024])
        green = np.array(test_data[b'data'][i][1024:2048])
        blue = np.array(test_data[b'data'][i][2048::])
        image_data[:,:,0]=red.reshape(imageShape)
        image_data[:, :, 1]=green.reshape(imageShape)
        image_data[:, :, 2]=blue.reshape(imageShape)

        fLabel=meta_data[b'fine_label_names'][test_data[b'fine_labels'][i]]
        fig.add_subplot(8,4,i+1,title=fLabel).imshow(image_data)
        print()
    plt.show()



    print(list(meta_data.keys()))
    print(meta_data[b'coarse_label_names'][10])





