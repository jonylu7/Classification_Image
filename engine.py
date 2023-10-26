from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_step(model:nn.Module,loss_fn:torch.nn,optimizer:torch.optim,train_data:DataLoader,device:torch.device):

    model.train()
    train_loss,train_acc=0,0

    for batch,(X,y) in train_data:
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
            X,y=X.to(device),y.to(device)
            y_pred_test=model(X)
            test_loss+=loss_fn(y_pred_test,y)
            ## waiting to be implemant
            test_acc+=1

    return test_loss,test_acc


def train(epoches:int,model:nn.Module,loss_fn:torch.nn,optimizer:torch.optim):



    for epoch in tqdm(range(epoches)):
        train_step()
        test_step()
        print(epoch)


if __name__=="__main__":