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
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 0
        num = targets.size(0) # number of batches
#         print(num)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
#         print(intersection)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1. - iou

def train(dataloader, model, loss_fn, optimizer):
    model = model.to("cuda")
    model.train()
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y, image) in enumerate(dataloader):
        X = X.float()
        y = y.float()
        X, y = X.to(device), y.to(device)
        
        X = X.requires_grad_()
        # Compute prediction error
        pred = model(X)
        pred = F.sigmoid(pred)
#         loss = loss_fn(pred, y)
#         print(pred.shape)
#         print(y.shape)

        pred = torch.reshape(pred, (1,256,256))
#         print(y.shape)
#         pred_flat = pred.view(-1)

#         y_flat = y.view(-1)
            
#         print(pred_flat.shape)
        loss = loss_fn((pred), y)
        
#         loss = loss_fn(pred, y)
#         print(loss)
        total_loss+=abs(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 25 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"total_loss: {total_loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss

def valid(dataloader, model, loss_fn, optimizer):
    model.eval()
    model = model.to("cpu")
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X, y, image) in enumerate(dataloader):
        X = X.float()
        y = y.float()
        pred = model(X)
        pred = F.sigmoid(pred)
        pred = torch.reshape(pred, (1,256,256))
        loss = loss_fn((pred), y)
        total_loss+=abs(loss)
    return total_loss


train_dataset = get_dataset("train")
val_dataset = get_dataset("val")
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
model = get_model()
loss_fn = IoULoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 500
device = 'cuda' if torch.cuda.is_available() else "cpu"
# device = 'cpu'
writer = SummaryWriter()
print(device)
min_loss=10000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    total_loss_train = train(train_dataloader, model, loss_fn, optimizer)
    print("training loss : ", total_loss_train/len(train_dataloader))
    writer.add_scalar("Loss/train", total_loss_train/len(train_dataloader), t)
    if(t%3==0): 
        with torch.no_grad():
            total_loss_val = valid(val_dataloader, model, loss_fn, optimizer)
            print("validation loss : ", total_loss_val/len(val_dataloader))
            writer.add_scalar("Loss/val", total_loss_val/len(val_dataloader), t)
            if(total_loss_val/len(val_dataloader) < min_loss ):
                min_loss = total_loss_val/len(val_dataloader)
                print("#########save_best_model###########")
                torch.save(model.state_dict(), "D:/Project/Tony/anemia/code/keras_version/best_model.pth")
print("Done!")
