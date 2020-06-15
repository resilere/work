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
# import torchvision
# import torch.nn as nn
import torch.optim as optim

from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)

N_EPOCH = 1
N_PATCH = 10
OUTPUT_FREQUENCY = 5
PATCH_SIZE = [32, 32, 32]
MIN_LOSS = 0.5
batch_size = 1


dir_charite = r"C:\Users\islere\Downloads\dicom_data\path_files_for_code\3d_model_orca.pth"
dir_home = "/home/eser/path_files_for_code/3d_model_orca.pth"
PATH = dir_home

charite_dir = Path(r"C:/Users/islere/Downloads/dicom_data/Training Set/")
home_dir = Path(r"/home/eser/Downloads/charite/orCaScore/Training Set/")

data_folder = home_dir

INPUT_FILES_TRAIN = (
    (
     str(data_folder / 'Images/TRV1P3CTI.mhd' ),
     str(data_folder / 'Reference standard/TRV1P3R.mhd')
     ),
    (
     str(data_folder / 'Images/TRV1P4CTI.mhd'),
     str( data_folder / 'Reference standard/TRV1P4R.mhd')

     )
    
   
)

INPUT_FILES_VALIDATION = (
   (

     str(data_folder / 'Images/TRV1P5CTI.mhd' ),
     str(data_folder / 'Reference standard/TRV1P5R.mhd' )

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
            
            
            output_array_max = torch.argmax(output_image.squeeze(), dim=0).detach().cpu().numpy()
            
            """here is the code for the patch plots"""
            dtp.plot_patches(output_array_max,label,input_image,patch_index,"coolwarm")


            train_loss = 0.0
    print('Finished Training')
    net.eval()
    
    for j, sample2 in enumerate(validation_loader, 0):
        
        input_image = sample2["image"].float()
        label = sample2["label"].long()
        patch_index = sample["patch_index"]
        
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
            
            """here is the code for the patch plots"""
            dtp.plot_patches(output_array_max,label,input_image,patch_index,"viridis")
           
            if valid_loss/OUTPUT_FREQUENCY < valid_loss_min:
                valid_loss_min = valid_loss/OUTPUT_FREQUENCY
                torch.save(net.state_dict(), PATH)
            valid_loss = 0.0


