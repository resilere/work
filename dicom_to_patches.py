# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:22:19 2020

@author: islere
"""
import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import dicom_lesen as dcr
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt

class data_patches(Dataset):
    
    def __init__(self, image_file_path, label_file_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.index_list = []
        self.patch_size = None
        self.image = dcr.nifti_to_array(image_file_path)[0] #ändern wenn image file ein dicom ist
        self.label = dcr.nifti_to_array(label_file_path)
        
    def threshold(self, array, threshold):
  
        mask = np.where((array <threshold), 0, 1)
        return mask
    
                
    def __len__(self):
        return len(self.index_list[0])#was kommt hier hin?
    
    def random_index(self,  patch_size,number_patches):
        image_shape = self.image.shape
        #print(image_shape)
        #print(self.label.shape)
        x_random = np.random.choice(image_shape[0]-patch_size[0]+1, number_patches)
        y_random = np.random.choice(image_shape[1]-patch_size[1]+1, number_patches)
        z_random = np.random.choice(image_shape[2]-patch_size[2]+1, number_patches)
        
        index_list = [x_random,y_random,z_random]
        self.index_list = index_list
        self.patch_size = patch_size


    def __getitem__(self, idx):
        image_patch = view_as_windows(self.image,self.patch_size)[self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx],:,:,:]
        label_patch = view_as_windows(self.label, self.patch_size)[self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx],:,:,:]
        sample = {"image":image_patch, "label":label_patch.astype(np.int_)}
        
        return sample
    

# =============================================================================
# data = data_patches(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_001.nii.gz', r'C:\Users\islere\Task01_BrainTumour\labelsTr\BRATS_001-labels.nii.gz')
# print(data.image.shape)
# print(data.label.shape)
# print(data.index_list)
# data.random_index( [1,128,128],10)  
# print(data.index_list)
# print(data[4:8]) 
# =============================================================================

###print(data.index_list)
#plt.imshow(data[5]["image"][0,:,:])
#plt.show()
#plt.close()
#plt.imshow(data[5]["label"][0,:,:])
#plt.show()    
##plt.close()
#plt.imshow(data[5]["label"].reshape([200,200]))
#plt.show() 
        
#plt.close() 
#plt.imshow(data[6]["image patch"].reshape([200,200]))
#plt.show()    
#plt.close()    
