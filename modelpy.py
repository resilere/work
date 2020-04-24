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
        #print('conv1',x.shape)
        y = x
        #print(x.shape)
        x = F.max_pool2d(x, (2,2))
        #print('maxpool', x.shape)
        x = F.relu(self.conv2(x))
        #print('conv2',x.shape)
        x = F.relu(self.upconv1(x))
        #print('upconv1',x.shape)
        x = torch.cat((x,y), 1)
        #print('cat',x.shape)
        x = self.conv3(x)
        #print('conv3',x.shape)
        
        return x
class Net2_5D(nn.Module):
    
    '''this is 2,5 D triplanar CNN model'''
    
    def __init__(self):
        super(Net2_5D, self).__init__()
        self.conv1 = nn.Conv2d(32, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        
        self.conv3 = nn.Conv2d(16, 32*4, 5, padding=2)

    def forward(self, x):
        print('x', x.shape)
        x = F.relu(self.conv1(x))
        print('conv1',x.shape)
        
# =============================================================================
#         x = F.max_pool2d(x, (2,2))
#         print('maxpool', x.shape)
# =============================================================================
        x = F.relu(self.conv2(x))
        print('conv2',x.shape)
        
        x = self.conv3(x)
        print('conv3',x.shape)
        
        return x
