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
        print('label shape', self.label.shape)
# =============================================================================
#         x_random = np.random.choice(range(30, 140-patch_size[0]), number_patches)
#         y_random = np.random.choice(range(55, 185-patch_size[1]), number_patches)
#         z_random = np.random.choice(range(50, 220-patch_size[2]), number_patches)
# =============================================================================
#        x_random = np.random.choice(image_shape[0]-patch_size[0]+1, number_patches)
#        y_random = np.random.choice(image_shape[1]-patch_size[1]+1, number_patches)
#        z_random = np.random.choice(image_shape[2]-patch_size[2]+1, number_patches)
        x_label_positive =[]
        y_label_positive =[]
        z_label_positive =[]
        
        for i in range(number_patches):
            x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
            y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
            z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
            while np.max(self.label[x_random[0],y_random[0]: y_random[0] + patch_size[1],z_random[0]:z_random[0]+ patch_size[2]]) == 0:
                x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
                y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
                z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
            x_label_positive.append(x_random[0])
            y_label_positive.append(y_random[0])
            z_label_positive.append(z_random[0])
        
        index_list = [x_label_positive,y_label_positive,z_label_positive]
        self.index_list = index_list
        self.patch_size = patch_size
        print('index list:', index_list)


    def __getitem__(self, idx):
        image_patch = view_as_windows(self.image,self.patch_size)[self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx],:,:,:]
        label_patch = view_as_windows(self.label, self.patch_size)[self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx],:,:,:]
        sample = {"image":image_patch, "label":label_patch.astype(np.int_)}
        
        return sample
    def crop_image_only_outside(self, tol=0):
    # img is 3D image data
    # tol  is tolerance
        #print(self.image.shape)
        img = self.image 
        
        mask = img>tol
        
        m,n,o = img.shape
        
        masko,maskm,maskn = mask.any((0,1)),mask.any((1,2)),mask.any((0,2))
        #print(masko)
        
        col_start,col_end = maskn.argmax(),n-maskn[::-1].argmax()
        
        row_start,row_end = maskm.argmax(),m-maskm[::-1].argmax()
        z_start, z_end = masko.argmax(),o-masko[::-1].argmax()
        self.image = img[row_start:row_end,col_start:col_end, z_start:z_end]
        self.label = self.label[row_start:row_end,col_start:col_end, z_start:z_end]
        
        

# =============================================================================
#data = data_patches(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_001.nii.gz', r'C:\Users\islere\Task01_BrainTumour\labelsTr\BRATS_001-labels.nii.gz')
#cropped_image = data.crop_image_only_outside()
#data.random_index([1,32,32],100) 
#print(data.label.shape)
#plt.imshow(data[0]["label"][0,:,:])
#print(cropped_image.shape)
#print(image[103, 52,120])
#plt.imshow(data.image[90,:,:])
#print('image after crop', data.image.shape)
## print(data.label.shape)

#print(data.index_list)
 
# print(data.index_list)
# print(data[4:8]) 
# =============================================================================

#print(data.index_list)
#plt.imshow(data[5]["image"][0,:,:])
#plt.show()
#plt.close()

#plt.show()    
##plt.close()
#plt.imshow(data[5]["label"].reshape([200,200]))
#plt.show() 
        
#plt.close() 
#plt.imshow(data[6]["image patch"].reshape([200,200]))
#plt.show()    
#plt.close()    
