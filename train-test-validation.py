# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:27:58 2020

@author: islere
"""
import modelpy as module
import numpy as np
import matplotlib.pyplot as plt
import dicom_to_patches as dtp
import torch
#import torchvision
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler


N_EPOCH = 2
N_PATCH = 10
OUTPUT_FREQUENCY = 5


INPUT_FILES = (
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_001.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_001-labels.nii.gz'
    ),
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_002.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_002-labels.nii.gz'
    ),
)

# Preprocess - concatenate datasets

datasets = []

for image_file, label_file in INPUT_FILES:

    data = dtp.data_patches(image_file, label_file)
    
    #validation_data= .....
    data.crop_image_only_outside()
    print("Loaded %s, image shape: %s"%(image_file, str(data.image.shape)))

    #import ipdb; ipdb.set_trace()

    data.random_index([1,32,32], N_PATCH)

    #print(data.index_list)
    #print(len(data))
    #print("Dicom geladen")
    # validation_split = .2
    #dataset_size = len(data.image)

    datasets.append(data)

data = ConcatDataset(datasets)
'''this part is for test and validation split'''
batch_size = 1
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                sampler=valid_sampler)

net = module.Net2()
net.train()


#train_loader = torch.utils.data.DataLoader(data, batch_size = 1) #this is old 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# =============================================================================
# '''this is for tensorboard '''
# writer = SummaryWriter('runs/brain_images')
# 
# def plot_classes_preds(net, images, labels):
#     fig = plt.figure(figsize=(12, 48))
#     for idx in np.arange(4):
#         
#         plt.imshow(images[0][idx].detach().numpy())
#         
#     return fig
# =============================================================================
    

'''this is where the training begins'''    
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
                  (epoch + 1, i + 1, train_loss / 2000))
            train_loss = 0.0
            #print(output_image.shape)
# =============================================================================
#   '''this is for tensorboard, which gives an error '''             
#             bilder_zusammen = torch.cat((200*torch.argmax(output_image,1).unsqueeze(1).float(), 200*label.float(), input_image.float()),3)
#             #print(bilder_zusammen.shape)
#             
#             img_grid = torchvision.utils.make_grid(bilder_zusammen.squeeze(1))
#            
#              # ...log the running loss
#             writer.add_scalar('training loss', running_loss / 1000, epoch * len(train_loader) + i)
#             # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
#             writer.add_figure('predictions vs. actuals', plot_classes_preds(net, output_image, label), global_step=epoch * len(train_loader) + i)
#             running_loss = 0.0
#             #write to tensorboard
#             writer.add_image(r'C:\Users\islere\Downloads\dicom_data\runs\brain_images\brain_images_trained' + str(i) + '_' + str(epoch), img_grid)
#             writer.add_figure('figures', plt.imshow(label),close = True)
#             writer.flush()
#             
# =============================================================================
            
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
            #plt.savefig("out/out-%05d.jpg"%(epoch))
    
    print('Finished Training')
    
    for i, sample in enumerate(validation_loader, 0):
        
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
        valid_loss += loss.item()
   
        if i % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    # print every OUTPUT_FREQUENCY mini-batches
            plt.clf()
            print('[%d, %5d] validation loss: %.3f' %
                  (epoch + 1, i + 1, valid_loss / 2000))
            valid_loss = 0.0
            
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




#img_grid = torchvision.utils.make_grid(output_image[:,0:3,...])

