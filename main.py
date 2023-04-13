from __future__ import print_function, division
import plotly.graph_objects as go
import numpy as np
import scipy.spatial.distance
import math
import random
import h5py
import json
import os
from torchvision import transforms, utils
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from progressbar import ProgressBar

from model import *
from losses import *
from partnet import *
from tqdm import tqdm
import os.path
from os import path
##conda activate blender
#!pip install wandb -qU
from tqdm import tqdm

import wandb
#wandb.login()

# 🐝 initialise a wandb run
#wandb.init(
#        project="chair_level3-0",
#        config={
#            "epochs": 55,
#            "lr": 1e-3,
#            })
    
# Copy your config 
config = wandb.config


def train(pointnet, train_loader, val_loade, optimizer,device,epochs=35, save=True):
    for epoch in tqdm(range(config.epochs)): 
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs,labels,gt_mask_pl,gt_valid_pl = data['image'].to(device),data['category'].to(device),data['masks'].to(device),data['valid'].to(device)
            optimizer.zero_grad()
            outputs ,mask_pred,end_points, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
            seg_loss ,end_points = get_seg_loss(outputs, labels, m3x3, m64x64,end_points)
            ins_loss, end_points = get_ins_loss(mask_pred, gt_mask_pl, gt_valid_pl, end_points)
            loss=ins_loss+seg_loss
            loss.backward()
            optimizer.step()
            
            seg_pred_id = torch.argmax(outputs, dim=1)
            acc=seg_pred_id == labels
            acc=acc.float()
            seg_acc = torch.mean(acc)
            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
                    print('[Train Epoch %d, Batch %d] Loss: %f = %f seg_loss (Seg Acc: %f) + %f (ins_loss)' \
                    % (epoch+1, i+1, loss, seg_loss, seg_acc, ins_loss))
                    wandb.log({"epoch": epoch, "seg_loss": seg_loss,"seg_acc":seg_acc,"ins_loss":ins_loss })
                    #torch.save(pointnet.state_dict(),"/content/gdrive/MyDrive/weights2/level3loss"+str(i)+".pth")
                    

        pointnet.eval()
        correct = total = 0
        mcorrect = mtotal = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for i,data in enumerate(val_loader,0):
                    inputs,labels,gt_mask_pl,gt_valid_pl = data['image'].to(device),data['category'].to(device),data['masks'].to(device),data['valid'].to(device)
                    outputs ,mask_pred,end_points, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) * labels.size(1) 
                    correct += (predicted == labels).sum().item()

                    seg_pred_id = torch.argmax(outputs, dim=1)
                    acc=seg_pred_id == labels
                    seg_acc = torch.mean(acc.float())
            val_acc = 100 * correct / total
            print('Valid accuracy: label %d %%  ' % val_acc)
            print('[Validation Epoch %03d, Batch %03d] Loss: %f = %f seg_loss (Seg Acc: %f) + %f (ins_loss)' \
                    % (epoch+1, i+1, loss, seg_loss, seg_acc, ins_loss))
            wandb.log({"epoch": epoch, "val_seg_loss": seg_loss,"val_seg_acc":seg_acc,"val_ins_loss":ins_loss })


        # save the model
        if epoch >20:
           torch.save(pointnet.state_dict(), "/home/fatma/Desktop/my\ work/weights/level3_loss"+str(epoch)+"_"+str(val_acc)+".pth")


def main():
     root_dir='/home/fatma/Desktop/ba/partnet/ins_seg_h5_for_detection/Chair-3/'
     train_dataset,val_dataset= load_datat(root_dir)

     train_loader = DataLoader(dataset=train_dataset, batch_size=4,drop_last = True)
     val_loader = DataLoader(dataset=val_dataset, batch_size=4,drop_last = True)
     pointnet = PointNetSeg()
     
     optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     print(device)
     pointnet.to(device);

     #train(pointnet, train_loader, val_loader,optimizer,device,  save=True)

     #load best weight
     path='/home/fatma/Desktop/my work/weights/level3loss54_62.663.pth'
     pointnet.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
     pointnet.eval()
     batch = next(iter(val_loader))
     
     #print_acc(batch,pointnet)
     myplot(batch)
    

if __name__ == "__main__":
    main()
