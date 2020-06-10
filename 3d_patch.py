# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:27:58 2020

@author: islere
"""
import modelpy as module
import numpy as np
import sys
import matplotlib.pyplot as plt
import dicom_to_patches as dtp
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(threshold=sys.maxsize)

N_EPOCH = 1
N_PATCH = 10
OUTPUT_FREQUENCY = 5
PATCH_SIZE = [32, 32, 32]
MIN_LOSS = 10
batch_size = 1
PATH = "/home/eser/path_files_for_code/3d_model_orca.pth"

INPUT_FILES_TRAIN = (
    (
     r'/home/eser/Downloads/charite/orCaScore/Training Set/Images/TRV1P3CTI.mhd',
     r'/home/eser/Downloads/charite/orCaScore/Training Set/Reference standard/TRV1P3R.mhd'
     ),
    (
     r'/home/eser/Downloads/charite/orCaScore/Training Set/Images/TRV1P4CTI.mhd',
     r'/home/eser/Downloads/charite/orCaScore/Training Set/Reference standard/TRV1P4R.mhd'
     )
    
   
)

INPUT_FILES_VALIDATION = (
   (
     r'/home/eser/Downloads/charite/orCaScore/Training Set/Images/TRV1P5CTI.mhd',
     r'/home/eser/Downloads/charite/orCaScore/Training Set/Reference standard/TRV1P5R.mhd'
     ), 
    
)

train_data = dtp.concat_datasets(INPUT_FILES_TRAIN, N_PATCH, PATCH_SIZE)
validation_data = dtp.concat_datasets(INPUT_FILES_VALIDATION, N_PATCH, PATCH_SIZE)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           )
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                               )

net = module.Net2_5D()

"""this code is to load a trained model"""
#net.load_state_dict(torch.load(PATH))

net.train()

#weights = torch.FloatTensor([0.5, 5.0])
#criterion = nn.CrossEntropyLoss(weight = weights)
criterion = module.DiceLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


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
        
        """ get the inputs; data is a list of [inputs, labels] """
        net.train()
        input_image = sample["image"].float()
        label = sample["label"].long()
        patch_index = sample["patch_index"]
        '''zero the parameter gradients'''
        
        optimizer.zero_grad()
        ''' here the model is used and viewed as 5 dimensional tensor '''
        output_image = net(input_image).view(batch_size, 2, 32, 32, 32)

        """this is to try dice loss function"""
        n = 2
        
        label_vector = torch.nn.functional.one_hot(label, n) # size=(4,7,n)
        label_vector = label_vector.permute(0, 4, 2, 3, 1)
        
        
        loss= criterion(output_image,label_vector)
        #print('conv1', net.conv1.weight.grad)
        loss.backward()
        #print('conv1,after backward', net.conv1.weight.grad)
        optimizer.step()
        
        '''print statistics'''
        train_loss += loss.item()
   
        if i % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    
            """# print every OUTPUT_FREQUENCY mini-batches"""
            plt.clf()
            print('[%d, %5d] train loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / OUTPUT_FREQUENCY))
            
            random_slice = np.random.randint(32)
            
            
            output_array_max = torch.argmax(output_image.squeeze(), dim=0).detach().cpu().numpy()
            
            
            """here is temporary code to show inout and output image patches"""
            slice_indices = np.arange(0, 9)
            for i in range(4):
                fig, axes = plt.subplots(nrows = 3, ncols = 8)
                fig.set_figheight(12)
                fig.set_figwidth(32)
                
                for ind in range(8):
                    output_slices = output_array_max[slice_indices[ind], :, :]
                    axes[0,ind].imshow(output_slices, cmap = 'coolwarm')
                        
                    axes[0, ind].axis('off')
                    
                    label_slices = label.squeeze()[slice_indices[ind], :,:]
                    axes[1,ind].imshow(label_slices, cmap = 'coolwarm')
                    axes[1, ind].axis('off')
                    
                    input_slices = input_image.squeeze()[slice_indices[ind], :, :]
                    axes[2,ind].imshow(input_slices, cmap = 'gray')
                        
                    #axes[2, ind].axis('off')
                    axes[2, ind].set_xlabel('%s' % patch_index)
                plt.show()
                slice_indices = slice_indices + 8
            

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

        
        """this is to try dice loss function"""
        n = 2
        
        label_vector = torch.nn.functional.one_hot(label, n) 
        label_vector = label_vector.permute(0, 4, 2, 3, 1)
        
        
        loss= criterion(output_image,label_vector)
        valid_loss += loss.item()
   
        if j % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    
            """ print every OUTPUT_FREQUENCY mini-batches"""
            plt.clf()
            print('[%d, %5d] validation loss: %.3f' %
                  (epoch + 1, j + 1, valid_loss /  OUTPUT_FREQUENCY))
            
            output_array_max =  torch.argmax(output_image.squeeze(), dim=0).detach().cpu().numpy()
            
            slice_indices = np.arange(0, 29, 4)
            for i in range(4):
                fig, axes = plt.subplots(nrows = 3, ncols = 8)
                fig.set_figheight(12)
                fig.set_figwidth(32)
                
           
                for ind in range(8):
                    output_slices = output_array_max[slice_indices[ind], :, :]
                    axes[0,ind].imshow(output_slices, cmap = 'viridis')
                    
                    axes[0, ind].axis('off')
                    label_slices = label.squeeze()[slice_indices[ind], :,:]
                    axes[1,ind].imshow(label_slices, cmap = 'viridis')
                    axes[1, ind].axis('off')
                    input_slices = input_image.squeeze()[slice_indices[ind], :, :]
                    axes[2,ind].imshow(input_slices, cmap = 'gray')
                    axes[2, ind].axis('off')
                    
                plt.show()
                slice_indices = slice_indices + 1
           
            if valid_loss/OUTPUT_FREQUENCY < valid_loss_min:
                valid_loss_min = valid_loss/OUTPUT_FREQUENCY
                torch.save(net.state_dict(), PATH)
            
            images_together2 = torch.cat((torch.argmax(output_image[:,:,np.random.randint(32), :,:],1).unsqueeze(1), label[:,np.random.randint(32), :,:].unsqueeze(1), input_image[:,np.random.randint(32), :,:].unsqueeze(1).long()),3)
            
            #img_grid2 = torchvision.utils.make_grid(images_together2.squeeze(1))
            
            writer.add_images('validation image' + str(i) + '_' + str(epoch),images_together2)
            
            writer.add_scalar('validation loss', valid_loss /  OUTPUT_FREQUENCY, epoch * len(validation_loader) + j)
            
            valid_loss = 0.0
            writer.close()



