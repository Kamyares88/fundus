#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:27:15 2023

@author: simpleai
"""

import os
import time
import tqdm
import torch
import math
import pandas as pd
import torch.nn as nn


from utils import assess_output

def train(model, dataloader_dict, criterion, optimizer, device, 
          checkpoint_path, train_val_info_path, num_epochs, vgg=False):
    # defining dictionaries for recording train/val loss/accuracy
    train_val_info = {
        'train_epoch': [],'train_loss':[], 'train_accuracy':[],
        'val_epoch': [],'val_loss':[], 'val_accuracy':[]
        }
    least_loss = 1e10 # defining a very high loss value for purpose of saving the model
    
    # epoch loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('_' * 10)
        since = time.time()
        
        # train/val loop
        for phase in ['train', 'val']:
            # definnig variables for tracking number of samples, number of correct predictions, and loss
            num_corrects, num_samples, running_loss = 0, 0, 0
            if phase == 'train':
                model.train()
            else:
                if vgg:
                    for m in model.modules():
                        for child in m.children():
                            if type(child)==nn.BatchNorm2d:
                                child.track_running_stats = False
                                child.running_mean = None
                                child.running_var = None
                model.eval()
            
            # minibatch loop
            with tqdm.tqdm(dataloader_dict[phase]) as tepoch:
                for index, (images, labels) in enumerate(tepoch):
                    # sending the inputs to device
                    images = images.type(torch.float32).to(device)#.type(torch.FloatTensor)
                    #images = images.type(torch.cuda.FloatTensor) #if device=='cuda' #else images.type(torch.FloatTensor)
                    labels = labels.to(device)
                    labels_arg = torch.argmax(labels, dim=1).type(torch.long)
                    
                    
                    # forward
                    # track history only if we are in train phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        
                        loss = criterion(outputs, labels_arg)
                        running_loss += loss.item()
                        # back propagation
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                    num_corrects, num_samples = assess_output(outputs, labels, num_corrects, num_samples)
            # recording train/val info
            running_loss = least_loss if math.isinf(running_loss) else running_loss
            train_val_info[f"{phase}_epoch"].append(epoch)
            train_val_info[f"{phase}_loss"].append(running_loss/num_samples)
            train_val_info[f"{phase}_accuracy"].append(num_corrects/num_samples)
            print(f"epoch {epoch+1} {phase} >>> loss: {(running_loss/num_samples):.3f}, accuracy: {(num_corrects/num_samples):.3f}")
            
            # saving the model
            if phase == 'val' and train_val_info[f"{phase}_loss"][-1]<least_loss:
                print(f"saving the best model in {checkpoint_path}")
                least_loss = train_val_info[f"{phase}_loss"][-1]
                torch.save(model.state_dict(), checkpoint_path)
        
    # at the end of the training: loading the best mdoel in order to return it
    model.load_state_dict(torch.load(checkpoint_path))
    
    # saving train/val info
    train_val_df = pd.DataFrame(train_val_info)
    train_val_df.to_csv(train_val_info_path)
    
    return model, train_val_df
                
                    
                    
            
    