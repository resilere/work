# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:27:58 2020

@author: islere
"""
import modelpy as model
import numpy as np
import matplotlib.pyplot as plt
import dicom_to_patches as dtp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

net = model.Net()
net.train()

data = dtp.data_patches(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_001.nii', r'C:\Users\islere\Task01_BrainTumour\labelsTr\BRATS_001-labels.nii')


data.random_index([1,128,128],1000)
print(len(data))
print("Dicom geladen")
validation_split = .2
dataset_size = len(data)

# =============================================================================
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# train_indices, val_indices = indices[split:], indices[:split]
# 
# 
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
# 
# =============================================================================
train_loader = torch.utils.data.DataLoader(data, batch_size = 1)

#validation_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle = False, num_workers = 2, sampler=valid_sampler)
#data.random_index( [1,32,32],50)
#output = model.net(torch.tensor(data[4]["image"]).unsqueeze(0).float())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, sample in enumerate(train_loader, 0):
  
        # get the inputs; data is a list of [inputs, labels]
        input_image = sample["image"].float()
        label = sample["label"].long()
        #print(inputs.shape)
        #print(type(inputs))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output_image = net(input_image)
#        print(output_image.shape)
#        print(label.shape)
        #print(torch.max(outputs))
        loss= criterion(output_image, label.squeeze(1) )
        
        loss.backward()
        optimizer.step()
        #print(loss)
        # print statistics
        running_loss += loss.item()
   
        if i % 40 == 39:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            output_array = output_image.detach().numpy()[:, ::-1, :, :]
            output_array_max = np.argmax(output_array, axis=0)
            label = label.detach().numpy()[:, ::-1, :, :]
            #input_array = inputs.detach().numpy()[:, ::-1, :, :]
            f, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(output_array_max.squeeze().squeeze(), cmap = 'gray')
            ax2.imshow(label.squeeze().squeeze(), cmap = 'gray')
            ax3.imshow(input_image.squeeze().squeeze(), cmap = 'gray')
            plt.show()
           
            #plt.imshow(inputs,cmap = 'gray')

print('Finished Training')