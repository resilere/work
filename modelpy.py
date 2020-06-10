# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:47:22 2020

@author: islere
"""
import torch
import torch.nn as nn
import torch.nn.functional as F




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 4, 5, padding=2) #output size is how many classes you want
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x

class Net2(nn.Module):
    '''this is the neural network with concatenating '''
    
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.upconv1 = nn.ConvTranspose2d(16, 8, 5,2, padding = 2, output_padding=1)
        self.conv3 = nn.Conv2d(16, 4, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        y = x
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.upconv1(x))
        x = torch.cat((x,y), 1)
        x = self.conv3(x)
        
        return x
class Net2_5D(nn.Module):
    
    '''this is 2,5 D triplanar CNN model'''
    
    def __init__(self):
        super(Net2_5D, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 80, 5, padding=2)
        
        self.conv3 = nn.Conv2d(80, 32*4, 5, padding=2)
        self.conv4 = nn.Conv2d(32*4, 32, 5, padding=2)
        self.conv5 = nn.Conv2d(3*32, 2*32, 5, padding = 2)
        """this is batch normalization, for 32 channels, implemented after 
        convolutional layers but before Relu, except the last layer"""
        self.m64 = nn.BatchNorm2d(64)
        self.m80 = nn.BatchNorm2d(80)
        self.m128 = nn.BatchNorm2d(128)
        self.m32 = nn.BatchNorm2d(32)
    def forward(self, x):
        """Here we take the permutations of the dimensions of the input patch and pass through CNN layers"""
        
        y = x
        z = y
        
        
        x = F.relu(self.m64(self.conv1(x)))
        x = F.relu(self.m80(self.conv2(x)))
        x = F.relu(self.m128(self.conv3(x)))
        x = self.m32(self.conv4(x))
        print('x', x.shape)
        y = y.permute(0, 2, 3, 1)
        y = F.relu(self.m64(self.conv1(y)))
        y = F.relu(self.m80(self.conv2(y)))
        y = F.relu(self.m128(self.conv3(y)))
        y = self.m32(self.conv4(y))
        y = y.permute(0,3,1,2)
        print('y', y.shape)
        z = z.permute(0, 3, 1, 2)
        z = F.relu(self.m64(self.conv1(z)))
        z = F.relu(self.m80(self.conv2(z)))
        z = F.relu(self.m128(self.conv3(z)))
        z = self.m32(self.conv4(z))
        z = z.permute(0,2,3,1)
        print('z', z.shape)
        x = torch.cat([x,y,z], 1)
        x = self.conv5(x)
        
        return x


class DiceLoss(nn.Module):
    """this is the working dice loss  fuction from kaggle"""
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth= 0.000001):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim = 1)       
        inputs = inputs[: , 1, : , : , : ]
        targets = targets[: , 1, : , : , : ]
        #import pdb;pdb.set_trace()
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.reshape(-1) #Ich habe hier von view geandert weil es ein error gibt
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/((inputs**2).sum() + (targets**2).sum() + smooth)  
        
        return 1 - dice