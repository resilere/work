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

data = dtp.data_patches(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_001.nii.gz', r'C:\Users\islere\Task01_BrainTumour\labelsTr\BRATS_001-labels.nii')
val_data = dtp.data_patches(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_002.nii.gz', r'C:\Users\islere\Task01_BrainTumour\labelsTr\BRATS_002-labels.nii')
#validation_data= .....
cropped_image = data.crop_image_only_outside()
cropped_image = val_data.crop_image_only_outside()
print(data.image.shape)
data.random_index([1,32,32],1000)
number_patches = 100
val_data.random_index([1,32,32],number_patches)
# =============================================================================
# This is for train test split sampling. We will use another patient data for validation.
#
#batch_size = 1
# validation_split = .2
# shuffle_dataset = True
# random_seed= 42
# 
# # Creating data indices for training and validation splits:
# dataset_size = len(data)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
# 
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
# =============================================================================

print(len(data))
print("Dicom geladen")

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
train_loader = torch.utils.data.DataLoader(data, batch_size=1)
validation_loader = torch.utils.data.DataLoader(val_data,batch_size=1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
writer = SummaryWriter('runs/brain_images')
def plot_classes_preds(net, images, labels):
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        
        plt.imshow(images[0][idx].detach().numpy())
        
    return fig
    
    
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    counter = 0

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
 
            print(len(list(validation_loader)))
            for sample_validation in list(validation_loader)[counter%(number_patches-2):counter%(number_patches-2) +2]:
                
  
                # get the inputs; data is a list of [inputs, labels]
                val_input_image = sample_validation["image"].float()
                val_label = sample_validation["label"].long()
                test_output = net(val_input_image)
                f, (ax1, ax2, ax3) = plt.subplots(1,3)
                ax1.imshow(np.argmax(test_output.detach().numpy()[0], axis =0), cmap = 'coolwarm')
                ax2.imshow(val_label.squeeze().squeeze(), cmap = 'coolwarm')
                ax3.imshow(val_input_image.squeeze().squeeze(), cmap = 'gray')
                plt.show()
                #to put in tensorboard
                val_bilder_zusammen = torch.cat((200*torch.argmax(test_output,1).unsqueeze(1).float(), 200*val_label.float(), val_input_image.float()),3)
                val_img_grid = torchvision.utils.make_grid(val_bilder_zusammen.squeeze(1))
                writer.add_image(r'C:\Users\islere\Downloads\dicom_data\runs\brain_images\brain_images_validated' + str(i) + '_' + str(epoch), val_img_grid)
            counter+=2
                
# =============================================================================
# PATH = './brain_images_net.pth'
# torch.save(net.state_dict(), PATH)
print('Finished Training')
# net.load_state_dict(torch.load(PATH))
# =============================================================================
# =============================================================================
#             validation_image = validation_data[?????]["image"]
#             validation_label = ....["label"]
#             net.eval()
#             validation_output=net(...)
#             loss---
#             tenorboard...
# =============================================================================
           
            #plt.imshow(inputs,cmap = 'gray')


#print(output_image.shape)
# tensorboard codes
#writer = SummaryWriter('runs/brain_images')
img_grid = torchvision.utils.make_grid(output_image[:,0:3,...])

# show images
#plt.imshow(img_grid, one_channel=True)

# write to tensorboard
#writer.add_image(r'C:\Users\islere\Downloads\dicom_data\runs\brain_images\brain_images_trained', img_grid)
#writer.flush()
#writer.close()
