# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:22:19 2020

@author: islere
"""

import numpy as np
from torch.utils.data import Dataset
import dicom_lesen as dcr
from skimage.util.shape import view_as_windows
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt 

class data_patches(Dataset):
    
    def __init__(self, image_file_path, label_file_path):
        """
        defines the inputs
        """
        self.index_list = []
        self.patch_size = None
        self.image = dcr.nifti_to_array(image_file_path) #ändern wenn image file ein dicom ist
        self.label = dcr.nifti_to_array(label_file_path)
        self.label = np.where(self.label > 0, 1, 0)
        
    
    def threshold(self, array, threshold):
  
        mask = np.where((array <threshold), 0, 1)
        return mask
    
                
    def __len__(self):
        return len(self.index_list[0])
    
    def random_index(self,  patch_size,number_patches):
        '''generates random indexes for one file, taking only the positive values in label file'''
        
        image_shape = self.image.shape
        if np.max(self.label) != 0: 
            """this condition is for the files that dont have any positive label"""
            x_label_positive =[]
            y_label_positive =[]
            z_label_positive =[]
            #print('the label has positive')
            for i in range(number_patches):
                x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
                y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
                z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
                #print('x, y, z rand', x_random, y_random, z_random)
                
                while np.max(self.label[x_random[0]: x_random[0] + patch_size[0],y_random[0]: y_random[0] + patch_size[1],z_random[0]:z_random[0]+ patch_size[2]]) == 0:
                    x_random = np.random.choice(image_shape[0]-patch_size[0]+1, 1)
                    y_random = np.random.choice(image_shape[1]-patch_size[1]+1, 1)
                    z_random = np.random.choice(image_shape[2]-patch_size[2]+1, 1)
                x_label_positive.append(x_random[0])
                y_label_positive.append(y_random[0])
                z_label_positive.append(z_random[0])
                index_list = [x_label_positive,y_label_positive,z_label_positive]
        else:  
            index_list = [np.random.randint(low=1, high=(image_shape[0]-patch_size[0]+1), size= number_patches), 
                              np.random.randint(low=1, high=(image_shape[1]-patch_size[1]+1), size= number_patches),
                              np.random.randint(low=1, high=(image_shape[2]-patch_size[2]+1), size= number_patches)
                              ]
            
        
        
        self.index_list = index_list
        self.patch_size = patch_size
        print('index list:', index_list)


    def __getitem__(self, idx):
        """this part gets the patches according to the patch size"""
        
        image_patch = view_as_windows(self.image,self.patch_size)[self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx],:,:,:]
        label_patch = view_as_windows(self.label, self.patch_size)[self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx],:,:,:]
        patch_index = [self.index_list[0][idx],self.index_list[1][idx],self.index_list[2][idx]]
        sample = {"image":image_patch, "label":label_patch.astype(np.int_), "patch_index" : patch_index}
        
        return sample
    def crop_image_only_outside(self, tol=0):
        ''' 
        img is 3D image data
        tol  is tolerance 
        this function crops the image with zero values on the outside
        '''
        img = self.image 
        print('shape image', img.shape)
        mask = img>tol
        
        m,n,o = img.shape
        
        masko,maskm,maskn = mask.any((0,1)),mask.any((1,2)),mask.any((0,2))
        
        
        col_start,col_end = maskn.argmax(),n-maskn[::-1].argmax()
        
        row_start,row_end = maskm.argmax(),m-maskm[::-1].argmax()
        z_start, z_end = masko.argmax(),o-masko[::-1].argmax()
        self.image = img[row_start:row_end,col_start:col_end, z_start:z_end]
        self.label = self.label[row_start:row_end,col_start:col_end, z_start:z_end]
        
#this function is not inside the class        
def concat_datasets(input_files_list, N_PATCH, PATCH_SIZE ):  
    '''concatenates multiple datasets into one dataset'''
    datasets= []
    for image_file, label_file in input_files_list:
        
        data = data_patches(image_file, label_file)
                
        data.crop_image_only_outside()
        print("Loaded %s, image shape: %s"%(image_file, str(data.image.shape)))
    
        data.random_index(PATCH_SIZE, N_PATCH)
        
        datasets.append(data)
    
    return ConcatDataset(datasets)
"""this function is for plotting the patches"""
def plot_patches (output_array_max, label, input_image, patch_index, colour, PATCH_SIZE):
    """here is a code to show inout and output image patches"""
    slice_indices = np.arange(0, 9)
    for i in range(int(PATCH_SIZE/8)):
        fig, axes = plt.subplots(nrows = 3, ncols = 8)
        fig.set_figheight(12)
        fig.set_figwidth(32)
                
        for ind in range(8):
            output_slices = output_array_max[slice_indices[ind], :, :]
            axes[0,ind].imshow(output_slices, cmap = colour)
            
            axes[0, ind].axis('off')
            label = label.cpu()
            print(label.device)
            label_slices = label.squeeze()[slice_indices[ind], :,:]
            axes[1,ind].imshow(label_slices, cmap = colour)
            axes[1, ind].axis('off')
            
            input_image=input_image.cpu()
            input_slices = input_image.squeeze()[slice_indices[ind], :, :]
            axes[2,ind].imshow(input_slices, cmap = 'gray')
            
            #axes[2, ind].axis('off')
            axes[2, ind].set_xlabel('%s' % patch_index)
        plt.show()
        slice_indices = slice_indices + 8
            
