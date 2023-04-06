#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:18:41 2023

@author: simpleai
"""
import os
import torch
import matplotlib.pyplot as plt


def img_viewer(
        img_list, title_list, image_title, image_title_pos_y=1.01,
        i=1, j=1, figsize=(4,4), axis=False, save=False, tight=True,
        hspace=None, wspace=None, save_path='~/subplot.png',
        panel_title_fontsize='medium', title_font_size='medium',
        show=True, cmap='gray', *args
        ):

    fig, ax = plt.subplots(i, j, figsize=figsize)
    if tight:
        plt.tight_layout()
    
    # if the subplot has only 1 panel
    if i==1 and j==1:
        ax.axis(axis)
        ax.imshow(img_list[0])
        ax.set_title(title_list[0], fontsize=panel_title_fontsize,
                     fontweight='bold')
        
    # if the subplot has 1 row or 1 column
    elif i==1 or j==1:
        ij = i if i>j else j
        for axi in range(ij):
            ax[axi].imshow(img_list[axi], cmap=cmap)
            ax[axi].axis(axis)
            ax[axi].set_title(title_list[axi], fontsize=panel_title_fontsize,
                              fontweight='bold')
            
    # if the subplot has more than 1 row and 1 column
    else:
        for axi in range(i):
            for axj in range(j):
                if axj < len(img_list[axi]):
                    ax[axi][axj].imshow(img_list[axi][axj], cmap=cmap)
                    ax[axi][axj].set_title(title_list[axi][axj],
                                           fontsize=panel_title_fontsize,
                                           fontweight='bold')
                ax[axi][axj].axis(axis)
    
    # setting title for the whole plot
    plt.suptitle(image_title, y=image_title_pos_y, fontsize=title_font_size,
                 fontweight='bold')
    if hspace:
        plt.subplots_adjust(hspace=hspace)
    if wspace:
        plt.subplots_adjust(wspace=wspace)
    if save:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    
# checking classification accuracy
def assess_output(y_hat, y, num_corrects, num_samples):    
    # evaluating correct predictions
    y_hat = y_hat.to('cpu')
    y = y.to('cpu')
    _, labels = y.max(1)
    _, preds = y_hat.max(1)
    num_corrects += (preds == labels).sum()
    num_samples += preds.size(0)
            
    # calculating num_correct predictions over total number of samples
    #acc = num_corrects / num_samples * 100
    #print(f"{mode} Accuracy: {num_corrects}/{num_samples} = {acc:.2f}")
    
    return int(num_corrects), int(num_samples)