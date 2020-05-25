# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:47:22 2020

@author: islere
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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
        self.conv1 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 80, 5, padding=2)
        
        self.conv3 = nn.Conv2d(80, 32*4, 5, padding=2)
        self.conv4 = nn.Conv2d(3*4*32, 2*32, 5, padding = 2)

    def forward(self, x):
        #print('x', x.shape)
        y = x
        z = y
        x = F.relu(self.conv1(x))
        #print('conv1',x.shape)
        
        x = F.relu(self.conv2(x))
        #print('conv2',x.shape)
        
        x = self.conv3(x)
        #print('conv3',x.shape)
        
        y = y.permute(0, 2, 3, 1)
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        y = y.permute(0,1,2,3)
        
        z = z.permute(0, 3, 1, 2)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = self.conv3(z)
        z = z.permute(0,1,2,3)
        
        x = torch.cat([x,y,z], 1)
        x = self.conv4(x)
        
        return x
class Net_new(nn.Module):
    """this is a new cnn to try triplanar cnn method"""
    def __init__(self):
        super(Net_new, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 4, 5, padding= 2)
        
    def forward(self, x):
        x1 = x[0] #yz plane
        x1 = F.relu(self.conv1(x1))
        
        x1 = F.max_pool2d(x1, (2,2))
        
        x1 = F.relu(self.conv2(x1))
        
        x1 = F.max_pool2d(x1, (2,2))
        
        x2 = x[1] #xz plane
        
        x2 = F.relu(self.conv1(x2))
        
        x2 = F.max_pool2d(x2, (2,2))
        
        x2 = F.relu(self.conv2(x2))
        
        x2 = F.max_pool2d(x2, (2,2))
        
        x3 = x[2] #xy plane
        
        x3 = F.relu(self.conv1(x3))
        
        x3 = F.max_pool2d(x3, (2,2))
        
        x3 = F.relu(self.conv2(x3))
        
        x3 = F.max_pool2d(x3, (2,2))
        
        merged = torch.cat([x1,x2,x3], 1)
        
        print('merged shape', merged.shape)
        return merged
def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total