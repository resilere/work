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
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
net = model.Net2()
net.train()

data = dtp.data_patches(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_001.nii.gz', r'C:\Users\islere\Task01_BrainTumour\labelsTr\BRATS_001-labels.nii.gz')
#validation_data= .....
cropped_image = data.crop_image_only_outside()
print(data.image.shape)
data.random_index([1,32,32],1000)
#print(data.index_list)
print(len(data))
print("Dicom geladen")
validation_split = .2
dataset_size = len(data.image)

# =============================================================================
# tensorboard --logdir=C:\Users\islere\Downloads\dicom_data\runs\brain_images
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
writer = SummaryWriter('runs/brain_images')
def plot_classes_preds(net, images, labels):
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        
        plt.imshow(images[0][idx].detach().numpy())
        
    return fig
    
    
for epoch in range(100):  # loop over the dataset multiple times

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
        
       
# =============================================================================
#         print(output_image.shape)
#         print(label.shape)
# =============================================================================
        #print(torch.max(outputs))
        loss= criterion(output_image, label.squeeze(0) )
        
        loss.backward()
        optimizer.step()
        #print(loss)
        # print statistics
        running_loss += loss.item()
   
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            #running_loss = 0.0
            print(output_image.shape)
            bilder_zusammen = torch.cat((200*torch.argmax(output_image,1).unsqueeze(1).float(), 200*label.float(), input_image.float()),3)
            print(bilder_zusammen.shape)
            
            img_grid = torchvision.utils.make_grid(bilder_zusammen.squeeze(1))
            #print(img_grid.shape)# tensorboard codes
            #img_grid_input = torchvision.utils.make_grid(torch.argmax(input_image,1))
            #img_grid_label = torchvision.utils.make_grid(torch.argmax(label,1))
            #img_grid = torchvision.utils.make_grid(torch.argmax(output_image,1))
            # show images
            #plt.imshow(img_grid, one_channel=True)
             # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_loader) + i)
            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, output_image, label),
                            global_step=epoch * len(train_loader) + i)
            running_loss = 0.0
            # write to tensorboard
            writer.add_image(r'C:\Users\islere\Downloads\dicom_data\runs\brain_images\brain_images_trained' + str(i) + '_' + str(epoch), img_grid)
            #writer.add_figure('figures', plt.imshow(label),close = True)
            #writer.flush()
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
            plt.show()
# =============================================================================
#             validation_image = validation_data[?????]["image"]
#             validation_label = ....["label"]
#             net.eval()
#             validation_output=net(...)
#             loas---
#             tenorboard...
# =============================================================================
           
            #plt.imshow(inputs,cmap = 'gray')

print('Finished Training')
print(output_image.shape)
# tensorboard codes
#writer = SummaryWriter('runs/brain_images')
img_grid = torchvision.utils.make_grid(output_image[:,0:3,...])

# show images
#plt.imshow(img_grid, one_channel=True)

# write to tensorboard
#writer.add_image(r'C:\Users\islere\Downloads\dicom_data\runs\brain_images\brain_images_trained', img_grid)
#writer.flush()
#writer.close()
