#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:40:40 2023

@author: Kamyar Esmaeili Pourfarhangi


In this "notebook" we will perform the following tasks:
0. data can be downloaded from https://ieee-dataport.s3.amazonaws.com/open/5172/A.%20RFMiD_All_Classes_Dataset.zip?response-content-disposition=attachment%3B%20filename%3D%22A.%20RFMiD_All_Classes_Dataset.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20230404%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230404T005022Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=ae7f8d7786b0629cd5f406d2d8954868d9c1aa840cbf4a64adebf9c12095e467
1. Download and unzip the data at a directory here named "path_to_data"
1. load the data csv and will create a metadata table from the provided labels and information
2. Randmoly select 50 images and visualize them
3. performing some descriptive statistical summary on the data

"""

# import packages
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# import custom functions
from utils.viewers import img_viewer

# data path lists
data_dir = '/Users/simpleai/Downloads/RFMiD_All_Classes_Dataset'
root_dir = '/Users/simpleai/Desktop/az_task/results/task1'
os.makedirs(root_dir, exist_ok=True)

data_type = ['Train', 'Val', 'Test']
image_path_dir_dict = {}
label_path_df = {}
global_df = pd.DataFrame() # this dataframe will comtain all the train/val/test labels
for data in data_type:
    image_path_dir_dict[data] = os.path.split(glob.glob(os.path.join(data_dir, f"*/*{data}*/*.png"))[0])[0]
    label_path_df[data] = pd.read_csv([i for i in glob.glob(os.path.join(data_dir, f"*/*{data}*.csv"))][0])
    
    # adding path to data to the csv file
    label_path_df[data]['image_path'] = [os.path.join(image_path_dir_dict[data],f"{i+1}.png") for i in label_path_df[data].index]
    
    # adding the data_type (train/val/test) to the label data
    label_path_df[data]['type'] = [data for i in label_path_df[data].index]
    global_df = pd.concat([global_df, label_path_df[data]], axis=0)
    global_df = global_df.reset_index(drop=True)

    # random visualization of 50 images from the train set
    img_list = list(np.random.randint(0,label_path_df[data].shape[0],50))
    
    # plotting 2 images each having a 5 by 5 (25) images
    if data=='Train':
        mia_dict = ['First', 'Second']
        for mia in range(2):
            img_arr_list = []
            img_title_list = []
            for i in range(5):
                img_row_list = []
                img_row_title_list = []
                for j in range(5):
                    # number of the image
                    ij = mia*25 + i*5 + j
                    # loading the image
                    img_row_list.append(np.asarray(Image.open(label_path_df[data].loc[img_list[ij],'image_path'])))
                    # making the label
                    img_title = ''
                    for col in [i for i in label_path_df[data].columns][1:-2]:
                        if label_path_df[data].loc[img_list[ij],col]==1 and col=='Disease_Risk':
                            img_title = f"{img_title}{col}\n"
                        elif label_path_df[data].loc[img_list[ij],col]==1:
                            img_title = f"{img_title}{col}_"
                    # labeling the images with no available label, based on the paper they are "healthy"
                    if img_title == '':
                        img_title = 'Healthy '
                    img_row_title_list.append(img_title[:-1])
                img_arr_list.append(img_row_list)
                img_title_list.append(img_row_title_list)
            
            #img_viewer(img_arr_list, img_title_list, f"{mia_dict[mia]} 25 images from '{data}' dataset",
            #           image_title_pos_y=1.05,
            #           i=5, j=5, figsize=(12,8), panel_title_fontsize='medium',
            #           title_font_size='x-large', save=True, 
            #           save_path=f"{root_dir}/mia_25_{mia}")
        
        
# give a summary statitistcs on the images from each conditions
# 1. barplots of number of images per category
fig, ax = plt.subplots(3,1,figsize=(12,10))
for d,data in enumerate(data_type):
    # curating data for barplot visualization
    metadata_df = label_path_df[data].iloc[:,1:-2]
    healthy_df = pd.DataFrame({'Healthy':[0 if metadata_df.loc[i,'Disease_Risk']==1 else 1 for i in metadata_df.index]})
    metadata_df = pd.concat([healthy_df, metadata_df], axis=1)
    sum_metadata_df = pd.DataFrame(metadata_df.sum(), columns=['num_cases'])
    sum_metadata_df['condition'] = [i for i in sum_metadata_df.index]
    
    # plotting the data
    sns.barplot(sum_metadata_df, x='condition', y='num_cases', ax=ax[d])
    
    # adjusting the plot for better visualization
    ax[d].set_ylim([0,sum_metadata_df['num_cases'].max()*1.1])
    xlabel = '' if not d==2 else 'condition'
    ax[d].set_xlabel(xlabel, fontsize='medium', fontweight='bold')
    ax[d].set_ylabel('number of images', fontsize='medium', fontweight='bold')
    ax[d].set_xticklabels(ax[d].get_xticklabels(), rotation=45, ha='right')
    ax[d].set_title(f"{data} data", fontsize='medium', fontweight='bold')
    for i in ax[d].containers:
        ax[d].bar_label(i,)
        
    plt.suptitle('distribution of image conditions', y=1.01, fontsize='large',
                 fontweight='bold')
    plt.tight_layout()
    
plt.savefig(f"{root_dir}/data_distribution.png", dpi=600, bbox_inches='tight')
plt.show()

# 2. selecting a binary labeled data (healthy, disease) for a binary classification task
# binary labels: healthy: 0; disease_risk: 1
binary_label_dict = {}
binary_label_dict['ID'] = [i for i in global_df['ID']]
binary_label_dict['Healthy'] = [1 if i==0 else 0 for i in global_df['Disease_Risk']]
binary_label_dict['Disease_Risk'] = [1 if i==1 else 0 for i in global_df['Disease_Risk']]
binary_label_dict['label'] = [i for i in global_df['Disease_Risk']]
binary_label_dict['image_path'] = [i for i in global_df['image_path']]
binary_label_dict['type'] = [i for i in global_df['type']]
binary_label_df = pd.DataFrame(binary_label_dict)
binary_label_df.to_csv(f"{root_dir}/binary_classification_metadata.csv")
# vizualizing number of selected data for binary classification
fig, ax = plt.subplots(1,3,figsize=(6,4))
ylim_max = 0
for d,data in enumerate(data_type):
    binary_label_df_type = binary_label_df.loc[binary_label_df['type']==data, ['Healthy', 'Disease_Risk']]
    sum_binary_label_df_type = pd.DataFrame(binary_label_df_type.sum(), columns=['num_classes'])
    sum_binary_label_df_type['condition'] = [i for i in sum_binary_label_df_type.index]
    # plotting the data
    sns.barplot(sum_binary_label_df_type, x='condition', y='num_classes', ax=ax[d])
    
    # adjusting the plot for better visualization
    ylim_max = max(sum_binary_label_df_type['num_classes'].max()*1.2, ylim_max)
    ax[d].set_ylim([0,ylim_max])
    ylabel = '' if not d==0 else 'number of images'
    ax[d].set_xlabel('condition', fontsize='medium', fontweight='bold')
    ax[d].set_ylabel(ylabel, fontsize='medium', fontweight='bold')
    ax[d].set_xticklabels(ax[d].get_xticklabels(), rotation=45, ha='right')
    ax[d].set_title(f"{data} data", fontsize='medium', fontweight='bold')
    for i in ax[d].containers:
        ax[d].bar_label(i,)
    plt.suptitle('distribution of binary labels', y=1.01, fontsize='large',
                 fontweight='bold')
    plt.tight_layout()
plt.savefig(f"{root_dir}/binary_data_distribution.png", dpi=600, bbox_inches='tight')
plt.show()

# 3. selecting a multiclass subset of data for classification
# note1: some images have belong to more than one classes. we will exclude them for simplicity
# note2: Based on the distribution acquired the following classes were selected for multi-class classification:
#    classes: healthy, DR, ARMD, MH, DN, TSLN, ODC
#    selection of these classes has been only based on the number of images per class existing in the data
multiclass_label_dict = {}
multiclass_label_dict['ID'] = [i for i in global_df['ID']]
multiclass_label_dict['image_path'] = [i for i in global_df['image_path']]
multiclass_label_dict['type'] = [i for i in global_df['type']]
multiclass_label_dict['Healthy'] = [1 if i==0 else 0 for i in global_df['Disease_Risk']]
multiclass_label_dict['label'] = [0 for i in global_df['image_path']]
for c,col in enumerate(['DR', 'ARMD', 'MH', 'DN', 'TSLN', 'ODC']):
    multiclass_label_dict[col] = [1 if i==1 else 0 for i in global_df[col]]
    
multiclass_label_df = pd.DataFrame(multiclass_label_dict)
# removing rows that belong to no class or more than 1 class
multiclass_label_df['sum']=[i for i in multiclass_label_df.drop(['image_path', 'ID', 'type'], axis=1).sum(axis=1)]
multiclass_label_df = multiclass_label_df.loc[multiclass_label_df['sum']==1, multiclass_label_df.columns!='sum']
multiclass_label_df = multiclass_label_df.reset_index(drop=True)
multiclass_label_df.to_csv(f"{root_dir}/multiclass_classification_metadata.csv")
# vizualizing number of selected data for binary classification
fig, ax = plt.subplots(1,3,figsize=(10,4))
ylim_max = 0
for d,data in enumerate(data_type):
    multiclass_label_df_type = multiclass_label_df.loc[multiclass_label_df['type']==data, ~multiclass_label_df.columns.isin(['ID', 'image_path', 'type'])]
    sum_multiclass_label_df_type = pd.DataFrame(multiclass_label_df_type.sum(), columns=['num_classes'])
    sum_multiclass_label_df_type['condition'] = [i for i in sum_multiclass_label_df_type.index]
    # plotting the data
    sns.barplot(sum_multiclass_label_df_type, x='condition', y='num_classes', ax=ax[d])
    
    # adjusting the plot for better visualization
    ylim_max = max(sum_multiclass_label_df_type['num_classes'].max()*1.2, ylim_max)
    ax[d].set_ylim([0,ylim_max])
    ylabel = '' if not d==0 else 'number of images'
    ax[d].set_xlabel('condition', fontsize='medium', fontweight='bold')
    ax[d].set_ylabel(ylabel, fontsize='medium', fontweight='bold')
    ax[d].set_xticklabels(ax[d].get_xticklabels(), rotation=45, ha='right')
    ax[d].set_title(f"{data} data", fontsize='medium', fontweight='bold')
    for i in ax[d].containers:
        ax[d].bar_label(i,)
    plt.suptitle('distribution of multiclass labels', y=1.01, fontsize='large',
                 fontweight='bold')
    plt.tight_layout()

plt.savefig(f"{root_dir}/multiclass_data_distribution.png", dpi=600, bbox_inches='tight')
plt.show()

# Do some quick feature extraction, filter analsysis to get some information and visualize them





















