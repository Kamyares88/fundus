#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:48:38 2023

@author: Kamyar Esmaeili Pourfarhangi
"""
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Fundus_dataset(Dataset): 
    def __init__(self, data_df, label_cols, transform=None):
        self.data_df = data_df
        self.label_cols = label_cols
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # open the image
        image = np.asarray(Image.open(self.data_df.loc[idx, 'image_path']))
        # centercropping the image centercrop size is dictated by the size of the largest image axes (i or j)
        j_radius = int(image.shape[1]/2)
        i_radius = int(image.shape[0]/2)
        if j_radius > i_radius:
            image = image[:,(j_radius-i_radius):(j_radius+i_radius),:]
        elif j_radius < i_radius:
            image = image[(i_radius-j_radius):(i_radius+j_radius),:,:]
        
        
        # label
        label = torch.tensor(list(self.data_df.loc[idx, self.label_cols])).float()
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    
def get_loader(data_df, label_cols, batch_size=16, transform=None, shuffle=True):
    fundus_dataset = Fundus_dataset(
        data_df = data_df, 
        label_cols = label_cols, 
        transform = transform
        )
    
    fundus_loader = DataLoader(
        dataset = fundus_dataset,
        batch_size = batch_size,
        shuffle = shuffle
        )
    
    return fundus_loader
        

