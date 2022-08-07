import torch
import torch.nn as nn
from Nail_dataset import get_dataset
from torch.utils.data import DataLoader
from Model import get_model
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import binary_cross_entropy
import torch
from torch.utils.tensorboard import SummaryWriter
from loss import *
from helper import *
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
resume = config["train"].getboolean("resume")
date = config["train"]["date"]
resume_model_path = config["train"]["resume_model_path"]
def train(dataloader, model, loss_fn, optimizer, criterion, writer, epoch):
    model = model.to("cuda")
#     model = model.to("cpu")
    model.train()
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    total_score = 0
    for batch, (X, y, image) in enumerate(dataloader):
        X = X.float()
        y = y.float()
        X, y = X.to(device), y.to(device)
        
        X = X.requires_grad_()
        # Compute prediction error
        pred = model(X)
        pred = F.sigmoid(pred)
        


        pred = pred.reshape(1,256,256)
        y = y.reshape(1,256,256)
        
        loss = loss_fn((pred), y)
        score = criterion((pred), y)

        total_loss+=abs(loss)
        total_score+=abs(score)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 25 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"total_loss: {total_loss:>7f}  [{current:>5d}/{size:>5d}]")
    writer.add_scalar("Loss/train", total_loss/len(dataloader), epoch)
    writer.add_scalar("IOU/train", total_score/len(dataloader), epoch)
    return total_loss


def valid(dataloader, model, loss_fn, optimizer, criterion, writer, epoch):
    model.eval()
    model = model.to("cpu")
    size = len(dataloader.dataset)
    total_loss = 0
    total_score = 0
    for batch, (X, y, image) in enumerate(dataloader):
        X = X.float()
        y = y.float()
        pred = model(X)
        pred = F.sigmoid(pred)
        pred = pred.reshape(1,256,256)
        y = y.reshape(1,256,256)
        loss = loss_fn((pred), y)
        score = criterion((pred), y)
        total_loss+=abs(loss)
        total_score+=abs(score)
    writer.add_scalar("Loss/val", total_loss/len(dataloader), epoch)
    writer.add_scalar("IOU/val", total_score/len(dataloader), epoch)
    return total_score



train_dataset = get_dataset("train", augment = True)
val_dataset = get_dataset("val", augment = True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
model = get_model()
if resume:
    model.load_state_dict(torch.load(resume_model_path))
    print("loading : ", resume_model_path)
#resume
# model.load_state_dict(torch.load("D:/Project/Tony/anemia/code/keras_version/models/best_model.pth"))



loss_fn = DiceLoss()
criterion = IoUScore()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 3000
device = 'cuda' if torch.cuda.is_available() else "cpu"
# device = "cpu"
writer = SummaryWriter("runs/"+date)
print(device)
max_score=0
print(date)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    total_loss_train = train(train_dataloader, model, loss_fn, optimizer, criterion, writer, t)
    print("training loss : ", total_loss_train/len(train_dataloader))
    if(t%20==0): 
        with torch.no_grad():
            total_score_val = valid(val_dataloader, model, loss_fn, optimizer, criterion, writer, t)
            print("validation score : ", total_score_val/len(val_dataloader))
            if(not os.path.exists("D:/Project/Tony/anemia/code/keras_version/models_"+date)):
                os.mkdir("D:/Project/Tony/anemia/code/keras_version/models_"+date)
            if(total_score_val/len(val_dataloader) > max_score ):
                max_score = total_score_val/len(val_dataloader)
                print("#########save_best_model###########")
                torch.save(model.state_dict(), "D:/Project/Tony/anemia/code/keras_version/models_"+date+"/best_model.pth")
            torch.save(model.state_dict(), "D:/Project/Tony/anemia/code/keras_version/models_"+date+"/model_"+str(t)+".pth")
print("Done!")