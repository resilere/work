# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:47:22 2020

@author: islere
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dicom_to_patches as dtp


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 4, 5, padding=2) #output size is how many classes you want
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x

#data = dtp.data_patches(r'C:\Users\islere\Downloads\dicom data\series-000001',3)
#data.random_index( [1,32,32],50)
#output = net(torch.tensor(data[4]["image"]).unsqueeze(0).float())
#print(output.shape)
#print(torch.max(output), torch.min(output))
#squeezed_output = output.squeeze(0)
#print(squeezed_output.shape)
#print(data[4]["image"].shape)

#loss_function = nn.MSELoss()

#loss = loss_function(squeezed_output, torch.tensor(data[4]["image"]))

#print(loss)