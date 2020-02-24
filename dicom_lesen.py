# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:46:29 2020

@author: islere
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import os
from skimage.util.shape import view_as_windows



# A file name that belongs to the series we want to read
#file_name = r'C:\Users\islere\Downloads\dicom data\series-000001\image-000001.dcm'
def dicom_to_array(directory):
    #data_directory =  r'C:\Users\islere\Downloads\dicom data\series-000001'
    
    # Read the file's meta-information without reading bulk pixel data
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    arr= sitk.GetArrayFromImage(image)
    #print(arr.shape)
    #print(arr[2, 3, 4])
    return arr
def nifti_to_array(directory):
    image = sitk.ReadImage(directory)
    array = sitk.GetArrayFromImage(image)
    return array

#array = nifti_to_array(r'C:\Users\islere\Downloads\dicom data\knochen.nii')
#print(array)

def dicom_visualize(directory, slicenumber):
    arr = dicom_to_array(directory)
   
    plt.imshow(arr[slicenumber,:,:], cmap = 'gray')
    plt.show()
    plt.close()
#dicom_visualize(r'C:\Users\islere\Downloads\dicom data\series-000001',40)

def dicom_nifti(directory, output):
    arr = dicom_to_array(directory)
    new_image = nib.Nifti1Image(arr, affine = np.eye(4))
    new_image.to_filename(os.path.join('build', output))
   

#dicom_nifti(r'C:\Users\islere\Downloads\dicom data\series-000001', r'C:\Users\islere\Downloads\dicom data\output_file.nii')

def threshold(array, threshold):
  
    mask = np.where((array <threshold), 0, 1)
    return mask
    
#treshold(r'C:\Users\islere\Downloads\dicom data\series-000001', 3)

def plot_masks (directory, schichten, thresholds):
    arr = dicom_to_array(directory)
    fig, ax = plt.subplots(len(schichten),len(thresholds))
    for i in range(len(thresholds)):
        for j in range(len(schichten)):
            ax[i,j].imshow(np.where(np.logical_and(arr[schichten[j],:,:]<thresholds[i][1],arr[schichten[j],:,:]>thresholds[i][0]),0,1), cmap ='gray')
    plt.show()
#plot_masks(r'C:\Users\islere\Downloads\dicom data\series-000001', [50,100,150],[[50,100],[300,400],[400,800]])

def patch_maker(directory, size):
    window_shape = (1, size[0], size[1])
    arr = dicom_to_array(directory)
    patch = view_as_windows(arr, window_shape)
    print(patch.shape)
#patch_maker(r'C:\Users\islere\Downloads\dicom data\series-000001', [32,32])
 
def patch_label_random (array_image, array_label):
    
    window_shape = (1,32,32)
    patches = view_as_windows(array_image, window_shape)
    indi_max=patches.shape
    x_random = np.random.randint(0,indi_max[0])
    y_random = np.random.randint(0,indi_max[1])
    z_random = np.random.randint(0,indi_max[2])
    image_patch = patches[x_random, y_random, z_random, 0,:,:]
    labels_patches = view_as_windows(array_label, window_shape)
    label_patch = labels_patches[x_random,y_random,z_random,0,:,:]
    return image_patch, label_patch
