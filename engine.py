from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_step(model:nn.Module,loss_fn:torch.nn,optimizer:torch.optim,train_data:DataLoader,device:torch.device):

    model.train()
    train_loss,train_acc=0,0
    for batch,(X,y) in (train_data):
        print(f"Train_Batch: {batch}")
        X,y=X.to(device),y.to(device)

        y_pred=model(X)
        loss = loss_fn(y, y_pred)
        train_loss+=loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        ## waiting to be implemant
        train_acc+=1

    return train_loss,train_acc




def test_step(model:nn.Module,loss_fn:torch.nn,test_data:DataLoader,device:torch.device):
    model.eval()
    test_loss,test_acc=0,0
    with torch.inference_mode():
        for batch,(X,y)in test_data:
            print(f"Test_Batch: {batch}")
            X,y=X.to(device),y.to(device)
            y_pred_test=model(X)
            test_loss+=loss_fn(y_pred_test,y)
            ## waiting to be implemant
            test_acc+=1


    return test_loss,test_acc


def train(epoches:int,model:nn.Module,loss_fn:torch.nn,optimizer:torch.optim,train_data:DataLoader,test_data:DataLoader,device:torch.device):

    for epoch in tqdm(range(epoches)):
        print(f"Epoch:{epoch}:")
        train_loss,train_acc=train_step(model,loss_fn,optimizer,train_data,device)
        test_loss,test_acc=test_step(model,loss_fn,test_data,device)
        print(f"Train_loss:{train_loss} | Test_loss:{test_loss} | Train_acc:{train_acc} | Test_acc:{test_acc}")


