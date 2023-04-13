from torch.utils.data import Dataset, DataLoader
import h5py
import json
from torchvision import transforms, utils
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split

root_dir='/home/fatma/Desktop/ba/partnet/ins_seg_h5_for_detection/Chair-1/'
#!pip install path.py;
from path import Path
import sys
sys.path.append(root_dir)


def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        gt_label = fin['gt_label'][:]
        gt_mask = fin['gt_mask'][:]
        gt_valid = fin['gt_valid'][:]
        gt_other_mask = fin['gt_other_mask'][:]
        return pts, gt_label, gt_mask, gt_valid#, gt_other_mask

def load_data(fn):
    #cur_json_fn = fn.replace('.h5', '.json')
    # record = load_json(cur_json_fn)
    pts, gt_label, gt_mask, gt_valid = load_h5(fn)
    return pts, gt_label, gt_mask, gt_valid#, gt_other_mask, record 



def load_datat(root_dir):
        #root_dir = "/content/gdrive/My Drive"
        train_h5_fn_list = []
        for item in os.listdir(root_dir):
            if item.endswith('.h5') and item.startswith('train-0'):
                train_h5_fn_list.append(item)
        val_h5_fn_list = []
        for item in os.listdir(root_dir):
            if item.endswith('.h5') and item.startswith('val-0'):
                val_h5_fn_list.append(item)
        print('train List: ', train_h5_fn_list)
        print('val List: ', val_h5_fn_list)
        
        path=train_h5_fn_list[0]
        cur_h5_fn = os.path.join(root_dir, path)
        pts, gt_label, gt_mask, gt_valid= load_data(cur_h5_fn)
        print("shape of pts: " ,pts.shape)
        
        i=0
        pts_total=pts        
        gt_label_total=gt_label
        gt_mask_total=gt_mask
        gt_valid_total=gt_valid
        #gt_other_mask_total=gt_other_mask
        for item in train_h5_fn_list:
                    if i>3 : 
                        print(item)
                        cur_h5_fn = os.path.join(root_dir, item)
                        pts, gt_label, gt_mask, gt_valid = load_data(cur_h5_fn)
                        pts_total=np.concatenate((pts_total, pts))
                        gt_label_total=np.concatenate((gt_label_total, gt_label))
                        gt_mask_total=np.concatenate((gt_mask_total, gt_mask))
                        gt_valid_total=np.concatenate((gt_valid_total, gt_valid))
                        #gt_other_mask_total= np.concatenate((gt_other_mask_total,gt_other_mask))
                        print("shape of pts_total: " ,pts_total.shape)
        
                    i+=1
        print("shape of pts: " ,pts_total.shape)
        
        pts =pts_total
        gt_label = gt_label_total
        gt_mask = gt_mask_total
        gt_valid = gt_valid_total
        #print(pts.shape)
        
        class Data(Dataset):
            """Face Landmarks dataset."""
            def __init__(self, pts,gt_label,gt_mask,gt_valid, transform=None):
                self.pts = pts
                self.gt_label = gt_label
                self.gt_mask = gt_mask
                self.gt_valid=gt_valid
               
            def __len__(self):
                return pts.shape[0]
        
            def __getitem__(self, idx):
                #if not self.valid:
                  #  theta = random.random()*360
                 #   image2 = utils.RandRotation_z()(utils.RandomNoise()(image2))  
                return {'image': np.array(pts[idx], dtype="float32"), 'category': gt_label[idx].astype(int) , 'masks':gt_mask[idx], 'valid':np.array(gt_valid[idx])}

        dset = Data(pts , gt_label, gt_mask, gt_valid)
        train_num = int(len(dset) * 0.95)
        val_num = int(len(dset) *0.05)
        if int(len(dset)) - train_num -  val_num >0 :
            train_num = train_num + 1
        elif int(len(dset)) - train_num -  val_num < 0:
            train_num = train_num -1
        #train_dataset, val_dataset = random_split(dset, [3000, 118])
        train_dataset, val_dataset = random_split(dset, [train_num, val_num])
        val_dataset.valid=True
        
        print('######### Dataset class created #########')
        print('Number of images: ', len(dset))
        print('Sample image shape: ', dset[0]['image'].shape)
        #print('Sample image points categories', dset[0]['category'], end='\n\n')
        return train_dataset,val_dataset