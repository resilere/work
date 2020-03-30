# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:22:19 2020

@author: islere
"""

import numpy as np
from torch.utils.data import Dataset
import dicom_lesen as dcr
from skimage.util.shape import view_as_windows

def random_index(image, label, patch_size, number_patches):
        '''generates random indexes for one file, taking only the positive values in label file'''
        
        image_shape = image.shape
               
        x_label_positive =[]
        y_label_positive =[]
        z_label_positive =[]
        
        for i in range(number_patches):
            x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
            y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
            z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
            while np.max(label[x_random[0],y_random[0]: y_random[0] + patch_size[1],z_random[0]:z_random[0]+ patch_size[2]]) == 0:
                x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
                y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
                z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
            x_label_positive.append(x_random[0])
            y_label_positive.append(y_random[0])
            z_label_positive.append(z_random[0])
        
        index_list = [x_label_positive,y_label_positive,z_label_positive]
        
        print('index list:', index_list)
class data_patches(Dataset):
    
    def __init__(self, image_file_path_list, label_file_path_list):
        """
        defines the inputs
        """
        self.index_lists = []
        self.patch_size = None
        self.images = dcr.multiple_nifti_arr(image_file_path_list) #ändern wenn image file ein dicom ist
        self.labels = dcr.multiple_nifti_arr(label_file_path_list)
        
    def threshold(self, array, threshold):
  
        mask = np.where((array <threshold), 0, 1)
        return mask
    
                
    def __len__(self):
        return len(self.index_lists[0])
    
    def multiple_random_index(self, patch_size, number_patches):
        '''generates random indexes for multiple files, taking only the positive values in label file'''
        for image in self.images:
            
            image_shape = image[0].shape
               
            x_label_positive =[]
            y_label_positive =[]
            z_label_positive =[]
            index_of_image = self.images.index(image)
            for i in range(number_patches):
                x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
                y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
                z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
                while np.max(self.labels[index_of_image][x_random[0],y_random[0]: y_random[0] + patch_size[1],z_random[0]:z_random[0]+ patch_size[2]]) == 0:
                    x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
                    y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
                    z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
                x_label_positive.append(x_random[0])
                y_label_positive.append(y_random[0])
                z_label_positive.append(z_random[0])
            
            index_list = [index_of_image, x_label_positive, y_label_positive, z_label_positive]
            self.index_lists.append(index_list)
            self.patch_size = patch_size
        print('index list:', self.index_lists)


    def __getitem__(self, idx):
        image_patch = view_as_windows(self.images,self.patch_size)[self.index_lists[0][idx],self.index_lists[1][idx],self.index_lists[2][idx],:,:,:]
        label_patch = view_as_windows(self.label, self.patch_size)[self.index_lists[0][idx],self.index_lists[1][idx],self.index_lists[2][idx],:,:,:]
        sample = {"image":image_patch, "label":label_patch.astype(np.int_)}
        
        return sample
    def crop_image_only_outside(self, tol=0):
        ''' 
        img is 3D image data
        tol  is tolerance 
        this function crops the image with zero values on the outside
        '''
        for image in self.images:
            img = image[0] 
            
            mask = img>tol
            print(img.shape)
            m,n,o = img.shape
            
            masko,maskm,maskn = mask.any((0,1)),mask.any((1,2)),mask.any((0,2))
            
            
            col_start,col_end = maskn.argmax(),n-maskn[::-1].argmax()
            
            row_start,row_end = maskm.argmax(),m-maskm[::-1].argmax()
            z_start, z_end = masko.argmax(),o-masko[::-1].argmax()
            image = img[row_start:row_end,col_start:col_end, z_start:z_end]
            return image
            for label in self.labels:
                label = label[row_start:row_end,col_start:col_end, z_start:z_end]
                return label
# =============================================================================
# after this is the code to try the multiple indexes        
# =============================================================================

list_of_image_directories = [r'/home/eser/Task01-BrainTumor/Images/BRATS_001.nii.gz','/home/eser/Task01-BrainTumor/Images/BRATS_002.nii.gz']

list_of_label_directories = [r'/home/eser/Task01-BrainTumor/Labels/BRATS_001-labels.nii','/home/eser/Task01-BrainTumor/Labels/BRATS_002-labels.nii']

data = data_patches(list_of_image_directories, list_of_label_directories)
cropped_image = data.crop_image_only_outside()

data.multiple_random_index([1,32,32],100)