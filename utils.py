import pickle
from pathlib import Path

def unpickle(file):
    with open(file,"rb") as f:
        dict=pickle.load(f,encoding="bytes")

        return dict

def readClasses(file:Path,superclass:bool=False)->list:
    meta_data = unpickle(file)
    if superclass:
        classes = list(str(i)[2:-1] for i in meta_data[b'coarse_label_names'])
    else:
        classes=list(str(i)[2:-1] for i in meta_data[b'fine_label_names'])

    return classes