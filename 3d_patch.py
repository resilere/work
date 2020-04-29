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
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data import ConcatDataset
#from torch.utils.data.sampler import SubsetRandomSampler


N_EPOCH = 1
N_PATCH = 10
PATCH_SIZE = [32, 32, 32]
OUTPUT_FREQUENCY = 5
MIN_LOSS = 10
batch_size = 1
PATH = "/home/eser/work/3d_model.pth"

INPUT_FILES_TRAIN = (
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_001.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_001.nii.gz'
    ),
          
    
)
# =============================================================================
# (
#         r'/home/eser/Task01-BrainTumor/Images/BRATS_002.nii.gz', 
#         r'/home/eser/Task01-BrainTumor/Labels/BRATS_002.nii.gz'
#     ),
#     (
#         r'/home/eser/Task01-BrainTumor/Images/BRATS_003.nii.gz', 
#         r'/home/eser/Task01-BrainTumor/Labels/BRATS_003.nii.gz'
#     ),
#     (
#         r'/home/eser/Task01-BrainTumor/Images/BRATS_004.nii.gz', 
#         r'/home/eser/Task01-BrainTumor/Labels/BRATS_004.nii.gz'
#     ),
#(
#        r'/home/eser/Task01-BrainTumor/Images/BRATS_006.nii.gz', 
#        r'/home/eser/Task01-BrainTumor/Labels/BRATS_006.nii.gz'
#    ),
# =============================================================================
INPUT_FILES_VALIDATION = (
    (
        r'/home/eser/Task01-BrainTumor/Images/BRATS_005.nii.gz', 
        r'/home/eser/Task01-BrainTumor/Labels/BRATS_005.nii.gz'
    ),
    
)

train_data = dtp.concat_datasets(INPUT_FILES_TRAIN, N_PATCH, PATCH_SIZE)
validation_data = dtp.concat_datasets(INPUT_FILES_VALIDATION, N_PATCH, PATCH_SIZE)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           )
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                               )

net = module.Net2_5D()
net.train()
#import ipdb; ipdb.set_trace()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

'''this is for tensorboard '''
writer = SummaryWriter('runs/brain_images')

def plot_classes_preds(net, images, labels):
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        
        plt.imshow(images[0][idx].detach().numpy())
        
    return fig
    

'''this is where the training begins''' 
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

        # forward + backward + optimize. 
        # @c: split the outputchannels in image direction in x -32- and segmentation classes -4-
        output_image = net(input_image).view(batch_size, 4,32,32,32)
        
        #import ipdb; ipdb.set_trace() @c: squeeze is shouldnt be neccesary anymore
        loss= criterion(output_image, label)
        
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
            print('label', label.shape)
            print('input',input_image.shape)
            plot_input = input_image.squeeze()[np.random.randint(32), :,:]
            plot_label = label.squeeze()[np.random.randint(32), :,:]
            plot_output = output_array_max[np.random.randint(32), :,:]
            #label = label.detach().numpy()[:, ::-1, :, :]
            #input_array = inputs.detach().numpy()[:, ::-1, :, :]
            f, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(plot_output , cmap = 'coolwarm')
            ax2.imshow(plot_label, cmap = 'coolwarm')
            ax3.imshow(plot_input, cmap = 'gray')
            plt.tight_layout()
            plt.show()
            #plt.savefig("out/out-%05d.jpg"%(epoch))
# '''this is for tensorboard, now it works ''' 

            #print('torch.argmax', torch.argmax(output_image[:,:,np.random.randint(32), :,:],1).unsqueeze(1).shape,label[:,np.random.randint(32), :,:].unsqueeze(1).shape,input_image[:,np.random.randint(32), :,:].unsqueeze(1).long().shape)
            images_together = torch.cat((torch.argmax(output_image[:,:,np.random.randint(32), :,:],1).unsqueeze(1), label[:,np.random.randint(32), :,:].unsqueeze(1), input_image[:,np.random.randint(32), :,:].unsqueeze(1).long()),3)
            #print('images together', images_together.shape)
            #img_grid = torchvision.utils.make_grid(images_together)
            writer.add_images('training image' + str(i) + '_' + str(epoch), images_together)
            # ...log the running loss
            writer.add_scalar('training loss', train_loss / OUTPUT_FREQUENCY, epoch *len(train_loader) + i)
            # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
            #writer.add_figure('predictions vs. actuals', plot_classes_preds(net, output_image, label), global_step=epoch * len(train_loader) + i)
            train_loss = 0.0
            #write to tensorboard
            
            # writer.add_figure('figures', plt.imshow(label),close = True)
            writer.close()
    
    print('Finished Training')
    net.eval()
    
    for j, sample2 in enumerate(validation_loader, 0):
        
        input_image = sample2["image"].float()
        label = sample2["label"].long()

        
        output_image = net(input_image).view(batch_size, 4,32,32,32)
        
        loss= criterion(output_image, label)
        valid_loss += loss.item()
   
        if j % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    # print every OUTPUT_FREQUENCY mini-batches
            plt.clf()
            print('[%d, %5d] validation loss: %.3f' %
                  (epoch + 1, j + 1, valid_loss /  OUTPUT_FREQUENCY))
            
                        
            output_array = output_image.detach().numpy()
            print(output_array.shape)
            output_array_max = np.argmax(output_array[0], axis=0)
            print(output_array_max.shape)
            plot_input = input_image.squeeze()[np.random.randint(32), :,:]
            plot_label = label.squeeze()[np.random.randint(32), :,:]
            plot_output = output_array_max[np.random.randint(32), :,:]
            f, (ax1, ax2, ax3) = plt.subplots(1,3)
            ax1.imshow(plot_output, cmap = 'viridis')
            ax2.imshow(plot_label, cmap = 'viridis')
            ax3.imshow(plot_input, cmap = 'gray')
            plt.tight_layout()
            plt.show()
            
            if valid_loss/OUTPUT_FREQUENCY < valid_loss_min:
                valid_loss_min = valid_loss/OUTPUT_FREQUENCY
                torch.save(net.state_dict(), PATH)
            
            images_together2 = torch.cat((torch.argmax(output_image[:,:,np.random.randint(32), :,:],1).unsqueeze(1), label[:,np.random.randint(32), :,:].unsqueeze(1), input_image[:,np.random.randint(32), :,:].unsqueeze(1).long()),3)
            
            #img_grid2 = torchvision.utils.make_grid(images_together2.squeeze(1))
            
            writer.add_images('validation image' + str(i) + '_' + str(epoch),images_together2)
            
            writer.add_scalar('validation loss', valid_loss /  OUTPUT_FREQUENCY, epoch * len(validation_loader) + j)
            
            valid_loss = 0.0
            writer.close()


# =============================================================================
# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
# 
# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
# 
# =============================================================================
            


