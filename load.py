#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:54:26 2020

@author: eser
"""

'''test for loading model'''
import modelpy as module
import numpy as np
import matplotlib.pyplot as plt
import dicom_to_patches as dtp
import torch
#import torchvision
import torch.nn as nn
import torch.optim as optim


N_EPOCH = 1
N_PATCH = 50
OUTPUT_FREQUENCY = 10
MIN_LOSS = 10
batch_size = 1
PATH = "/home/eser/work/lstmmodelgpu.pth"
PATH2 = "/home/eser/work/newmodelgpu.pth"

INPUT_FILES_TRAIN = (
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_001.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_001.nii.gz'
    ),
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_002.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_002.nii.gz'
    ),
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_003.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_003.nii.gz'
    ),
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_004.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_004.nii.gz'
    ),
       
    
)

INPUT_FILES_VALIDATION = (
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_005.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_005.nii.gz'
    ),
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_006.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_006.nii.gz'
    ),
)

train_data = dtp.concat_datasets(INPUT_FILES_TRAIN, N_PATCH)
validation_data = dtp.concat_datasets(INPUT_FILES_VALIDATION, N_PATCH)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,)

net = module.Net2()  
#net = model.load_state_dict(torch.load(PATH))

checkpoint = torch.load(PATH)
try:
    checkpoint.eval()
except AttributeError as error:
    print(error)
### 'dict' object has no attribute 'eval'

net.load_state_dict(checkpoint)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

valid_loss_min = MIN_LOSS   
for epoch in range(N_EPOCH):  # loop over the dataset multiple times

    train_loss = 0.0
    valid_loss = 0.0
    
    for i, sample in enumerate(train_loader, 0):
  
        # get the inputs; data is a list of [inputs, labels]
        input_image = sample["image"].float()
        label = sample["label"].long()
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output_image = net(input_image)
        
      
        loss= criterion(output_image, label.squeeze(0) )
        
        loss.backward()
        optimizer.step()
        
        # print statistics
        train_loss += loss.item()
   
        if i % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    # print every OUTPUT_FREQUENCY mini-batches
            plt.clf()
            print('[%d, %5d] train loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / OUTPUT_FREQUENCY))
            
            
            
            output_array = output_image.detach().numpy()
            print(output_array.shape)
            output_array_max = np.argmax(output_array[0], axis=0)
            print(output_array_max.shape)
            label = label.detach().numpy()[:, ::-1, :, :]
            #input_array = inputs.detach().numpy()[:, ::-1, :, :]
            f, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(output_array_max, cmap = 'coolwarm')
            ax2.imshow(label.squeeze().squeeze(), cmap = 'coolwarm')
            ax3.imshow(input_image.squeeze().squeeze(), cmap = 'gray')
            plt.tight_layout()
            plt.show()
            

            train_loss = 0.0
         
    
    print('Finished Training')
    net.eval()
    
    for j, sample2 in enumerate(validation_loader, 0):
        
        input_image = sample2["image"].float()
        label = sample2["label"].long()

        output_image = net(input_image)
        
        loss= criterion(output_image, label.squeeze(0)) 
        valid_loss += loss.item()
   
        if j % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    # print every OUTPUT_FREQUENCY mini-batches
            plt.clf()
            print('[%d, %5d] validation loss: %.3f' %
                  (epoch + 1, j + 1, valid_loss /  OUTPUT_FREQUENCY))
            
                        
            output_array = output_image.detach().numpy()
            print(output_array.shape)
            output_array_max = np.argmax(output_array[0], axis=0)
            print(output_array_max.shape)
            label = label.detach().numpy()[:, ::-1, :, :]
            #input_array = inputs.detach().numpy()[:, ::-1, :, :]
            f, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(output_array_max, cmap = 'viridis')
            ax2.imshow(label.squeeze().squeeze(), cmap = 'viridis')
            ax3.imshow(input_image.squeeze().squeeze(), cmap = 'gray')
            plt.tight_layout()
            plt.show()
            
            if valid_loss/OUTPUT_FREQUENCY < valid_loss_min:
                valid_loss_min = valid_loss/OUTPUT_FREQUENCY
                torch.save(net.state_dict(), PATH2)
                
            


