# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:47:22 2020

@author: islere
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import datetime



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
        self.conv0 = nn.Conv2d(16, 32, 5, padding = 2 )
        self.conv1 = nn.Conv2d(32, 64, 5, padding=2)
        
        self.conv2 = nn.Conv2d(64, 32*4, 5, padding=2)
        
        self.conv3 = nn.Conv2d(32*4, 32*8, 5, padding=2)
        
        self.upconv1 = nn.ConvTranspose2d(32*8, 32*8 , 5 , stride = 2, padding = 2, output_padding=1)
        
        self.conv4 = nn.Conv2d(32*8 + 32*4, 32*4, 5, padding =2 )
        
        self.upconv2 = nn.ConvTranspose2d(32*4, 32*4 , 5 , stride = 2, padding = 2, output_padding=1)
        
        self.conv5 = nn.Conv2d(32*2 + 32*4, 32*2, 5, padding =2 )
        
        self.conv6 = nn.Conv2d(3*2*32, 2*32, 5, padding = 2)
        
        self.conv7 = nn.Conv2d(2*32, 32, 5, padding = 2)
        """this is batch normalization, for 32 channels, implemented after 
        convolutional layers but before Relu, except the last layer"""
        self.m32 = nn.BatchNorm2d(32)
        self.m64 = nn.BatchNorm2d(64)
        self.m128 = nn.BatchNorm2d(128)
        self.m256 = nn.BatchNorm2d(256)
        #self.m32 = nn.BatchNorm2d(32)
    def forward(self, x):
        """Here we take the permutations of the dimensions of the input patch and pass through CNN layers"""
        planes = []
        
# =============================================================================
#         y = z = x
#         
#         y = y.permute(0, 2, 3, 1)
#         z = z.permute(0, 3, 1, 2)
#         planes.append(y)
#         planes.append(z)
#         
# =============================================================================
        planes.append(x)
        
        
        concat = []
        '''this is a loop to implement the same cnn layers for the different planes of the 3d patch'''
        for x in planes:
            x = F.relu(self.m32(self.conv0(x)))
            x = F.relu(self.m64(self.conv1(x)))
            x1 = x
            
            x = F.max_pool2d(x, (2,2))
            
            x = F.relu(self.m128(self.conv2(x)))
            x2 = x
            
            x = F.max_pool2d(x, (2,2))
            
            x = F.relu(self.m256(self.conv3(x)))
            
            x = F.relu(self.upconv1(x))
            #import pdb; pdb.set_trace()
            x = torch.cat((x, x2), 1)
            
            x = F.relu(self.m128(self.conv4(x)))
            
            x = F.relu(self.upconv2(x))
            
            x = torch.cat((x, x1), 1)
            
            x = F.relu(self.m64(self.conv5(x)))
            x = self.conv7(x)
            #concat.append(x)

# =============================================================================
#         x = torch.cat(concat, 1)
#         x = self.conv6(x)
#         
# =============================================================================
        return x


class DiceLoss(nn.Module):
    """this is the working dice loss  fuction from kaggle"""
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth= 0.000001):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim = 1)       

        inputs1 = inputs[: , 1, : , : , : ]
        targets1 = targets[: , 1, : , : , : ]
        inputs2 = inputs[:,0, : , : , : ]
        targets2 = targets[:,0, : , : , : ]

      
        #import pdb;pdb.set_trace()
        #flatten label and prediction tensors
        inputs1 = inputs1.view(-1)
        targets1 = targets1.reshape(-1) #Ich habe hier von view geandert weil es ein error gibt
        
        intersection1 = (inputs1 * targets1).sum()                            
        dice1 = (2.*intersection1 + smooth)/((inputs1**2).sum() + (targets1**2).sum() + smooth) 
        inputs2 = inputs2.view(-1)
        targets2 = targets2.reshape(-1) #Ich habe hier von view geandert weil es ein error gibt
        
        intersection2 = (inputs2 * targets2).sum()                            
        dice2 = (2.*intersection2 + smooth)/((inputs2**2).sum() + (targets2**2).sum() + smooth) 
        #print("dice1, dice2", dice1, dice2)
        return 1 - dice1 -0.3*dice2


def weights_to_list (list_of_conv, list_of_info, epoch, i, loss, OUTPUT_FREQUENCY):
    """this is a function to write the weights and grads of the conv layers in a list"""
    for conv in list_of_conv :
        #print('%d layer ' %list_of_conv.index(conv) , torch.max(conv.weight.grad))
        list_of_info.append(['[%d, %5d] ' % (epoch + 1, i + 1),
                                     "%.3f" % (loss / OUTPUT_FREQUENCY),
                                    'layer %d' %list_of_conv.index(conv), 
                                    "%.5f" %torch.max(conv.weight.grad).item(),
                                    "%.5f" %torch.min(conv.weight.grad).item(),
                                    "%.5f" %torch.max(conv.weight).item(),
                                    "%.5f" %torch.min(conv.weight).item()])
    return list_of_info

def list_to_excel(list_of_info) :
    """this is a function to save the list generated from weights_to_list into excel"""
    list_of_columns = ['Epoch_number', 'Loss', 'Conv_layer_num' , 'Grad_max ', 'Grad_min ', 'Weight_max', 'Weight_min']
    df = pd.DataFrame(data = list_of_info, columns = list_of_columns)
    print(df.head())
    time = datetime.datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
    df.to_excel(r'created_at_%s.xlsx'%(time) ,sheet_name='parameters')
#list_to_excel(list_of_info)
