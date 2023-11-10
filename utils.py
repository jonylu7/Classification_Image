import pickle
from pathlib import Path
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict

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

def save_model(model:torch.nn.Module,target_dir:str,model_name:str):
    target_dir_path=Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_pth=target_dir_path/model_name

    print(f"Save {model_name} at {target_dir_path}")
    torch.save(model.state_dict(),f=model_save_pth)

def save_results(results:Dict,targ_dict:str,fileName):
    targ_dict=Path(targ_dict)
    targ_dict.mkdir(parents=True,exist_ok=True)

    with open(targ_dict/fileName,"wb") as file:
        pickle.dump(results,file)
        print(f"dictionary saved to {Path(targ_dict)/fileName}")

def read_results(targ_dict:str,fileName:str):
    file_path=Path(targ_dict)/fileName
    with open(file_path,"rb") as file:
        results=pickle.load(file)
        print("dictionary loaded")
    return results


def display_results(results:Dict):
    print()
    width=range(len(results["train_loss"]))
    plt.figure(figsize=(15,7))
    plt.subplot(2,1,1)
    plt.plot(width,list(map(float,results["train_loss"])),label="train_loss")
    plt.plot(width,list(map(float,results["test_loss"])),label="test_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(width,list(map(float,results["train_acc"])),label="train_acc")
    plt.plot(width,list(map(float,results["test_acc"])),label="test_acc")
    plt.title("Acc")
    plt.legend()

    plt.show()