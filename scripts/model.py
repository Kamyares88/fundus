#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:09:03 2023

@author: Kamyar Esmaeili Pourfarhangi
"""

import os
import pandas as pd
import numpy as np
import tifffile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F 



class MyVGG16(nn.Module):
    def __init__(self, input_width=512, input_height=512, input_num_channels=3, 
                 num_classes=2, filter_num_list=[64, 128, 256, 512, 512]):
        super(MyVGG16, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_num_channels = input_num_channels
        self.num_classes = num_classes
        self.filter_num_list = filter_num_list
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(self.input_num_channels, self.filter_num_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[0]),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[0], self.filter_num_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[0]),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer21 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[0], self.filter_num_list[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[1]),
            nn.ReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[1], self.filter_num_list[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer31 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[1], self.filter_num_list[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[2]),
            nn.ReLU())
        self.layer32 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[2], self.filter_num_list[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[2]),
            nn.ReLU())
        self.layer33 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[2], self.filter_num_list[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer41 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[2], self.filter_num_list[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[3]),
            nn.ReLU())
        self.layer42 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[3], self.filter_num_list[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[3]),
            nn.ReLU())
        self.layer43 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[3], self.filter_num_list[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer51 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[3], self.filter_num_list[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[4]),
            nn.ReLU())
        self.layer52 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[4], self.filter_num_list[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[4]),
            nn.ReLU())
        self.layer53 = nn.Sequential(
            nn.Conv2d(self.filter_num_list[4], self.filter_num_list[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.filter_num_list[4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(int(self.input_width/32)*int(self.input_height/32)*self.filter_num_list[4], 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        #self.sig = nn.Sigmoid()
    
    def forward(self, x):
        y = self.layer11(x)
        y = self.layer12(y)
        
        y = self.layer21(y)
        y = self.layer22(y)
        
        y = self.layer31(y)
        y = self.layer32(y)
        y = self.layer33(y)
        
        y = self.layer41(y)
        y = self.layer42(y)
        y = self.layer43(y)
        
        y = self.layer51(y)
        y = self.layer52(y)
        y = self.layer53(y)
        
        y = y.reshape(y.size(0), -1)
        y = self.fc(y)
        y = self.fc1(y)
        y = self.fc2(y)
        #y = self.sig(y)
        
        return y
