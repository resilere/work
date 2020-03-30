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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset

N_EPOCH = 100
N_PATCH = 100
OUTPUT_FREQUENCY = 50


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

#import ipdb; ipdb.set_trace()

net = model.Net2()
net.train()


train_loader = torch.utils.data.DataLoader(data, batch_size = 1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
writer = SummaryWriter('runs/brain_images')

def plot_classes_preds(net, images, labels):
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        
        plt.imshow(images[0][idx].detach().numpy())
        
    return fig
    
    
for epoch in range(N_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0

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
        running_loss += loss.item()
   
        if i % OUTPUT_FREQUENCY == OUTPUT_FREQUENCY - 1:    # print every 2000 mini-batches
            plt.clf()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            #running_loss = 0.0
            print(output_image.shape)
            bilder_zusammen = torch.cat((200*torch.argmax(output_image,1).unsqueeze(1).float(), 200*label.float(), input_image.float()),3)
            print(bilder_zusammen.shape)
            
            img_grid = torchvision.utils.make_grid(bilder_zusammen.squeeze(1))
           
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
            plt.tight_layout()
            plt.savefig("out/out-%05d.jpg"%(epoch))

    

print('Finished Training')
print(output_image.shape)

#img_grid = torchvision.utils.make_grid(output_image[:,0:3,...])

