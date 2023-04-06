#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 23:49:03 2023

@author: Kamyar Esmaeili Pourfarhangi
"""
import os
os.chdir('/Users/simpleai/Desktop/az_task/scripts/')
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn

from dataset import get_loader
from model import MyVGG16
from train import train

# opening the label_df where both labels and path to images are located in
label_df_path = '/Users/simpleai/Desktop/az_task/results/task1/binary_classification_metadata.csv'
label_df = pd.read_csv(label_df_path, index_col=0)

# dividing the label_df into train, val, test dataset
train_df = label_df.loc[label_df['type']=='Train', :] #1519 Disease; 401 Healthy 0.8:0.2 imbalance
train_df = train_df.reset_index(drop=True)

val_df = label_df.loc[label_df['type']=='Val', :]
val_df = val_df.reset_index(drop=True)

test_df = label_df.loc[label_df['type']=='Test', :]
test_df = test_df.reset_index(drop=True)


# defining the transforms to be applied to the images
# transform will include on-the-fly augmentations later
transform = transforms.Compose([
   transforms.ToTensor(), 
   transforms.Resize((256,256)),
   transforms.Normalize(
       mean=[0.4914, 0.4822, 0.4465], 
       std=[0.2023, 0.1994, 0.2010]
       )
   ])

# hyperparameters
NUM_CLASSES = 2
NUM_EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1E-3
CLASS_WEIGHT = torch.tensor([0.8,0.2])
device = "mps" if torch.backends.mps.is_available() else "cpu" # I'm using an M1 chip which can perform vectorized calculations over Mac GPU using ARM technology


# defining the dataloader dictionary
dataloader_dict = {
    'train': get_loader(
        data_df = train_df, 
        label_cols = ['Healthy', 'Disease_Risk'],
        batch_size = BATCH_SIZE, 
        transform = transform, 
        shuffle = True),
    'val': get_loader(
        data_df = val_df, 
        label_cols = ['Healthy', 'Disease_Risk'],
        batch_size = BATCH_SIZE, 
        transform = transform, 
        shuffle = False),
    }

# defining the model
model = MyVGG16(num_classes=NUM_CLASSES, input_height=256, input_width=256).to(device)


# defining criterion, optimizer
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# training the model
model, train_val_df = train(
    model, 
    dataloader_dict, 
    criterion, 
    optimizer, 
    device, 
    checkpoint_path='/Users/simpleai/Desktop/az_task/results/task2/binary_classifier.pth', 
    train_val_info_path='/Users/simpleai/Desktop/az_task/results/task2/binary_classifier_info.csv',
    num_epochs=NUM_EPOCHS
    )

