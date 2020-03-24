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


def dicom_to_array(directory):
    '''turns dicom directories to arrays''' 
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    arr= sitk.GetArrayFromImage(image)
    
    return arr
list_arr = []
def multiple_nifti_arr(list):
    '''creates a list of the arrays from a list of nifti files'''
    
    for directory in list:
        image = sitk.ReadImage(directory)
        array = sitk.GetArrayFromImage(image)
        list_arr.append(array)
    return list_arr
        

def nifti_to_array(directory):
    '''turns nifti directories to arrays'''
    
    image = sitk.ReadImage(directory)
    array = sitk.GetArrayFromImage(image)
    return array

def dicom_visualize(directory, slicenumber):
    '''to visualize the slicenumber of a dicom'''
    
    arr = dicom_to_array(directory)
   
    plt.imshow(arr[slicenumber,:,:], cmap = 'gray')
    plt.show()
    plt.close()

def dicom_nifti(directory, output):
    '''from dicom directory to nifti image'''
    
    arr = dicom_to_array(directory)
    new_image = nib.Nifti1Image(arr, affine = np.eye(4))
    new_image.to_filename(os.path.join('build', output))
   

def threshold(array, threshold):
    '''threshold for a mask'''
    
    mask = np.where((array <threshold), 0, 1)
    return mask
    

def plot_masks (directory, schichten, thresholds):
    '''to plot the masks'''
    
    arr = dicom_to_array(directory)
    fig, ax = plt.subplots(len(schichten),len(thresholds))
    for i in range(len(thresholds)):
        for j in range(len(schichten)):
            ax[i,j].imshow(np.where(np.logical_and(arr[schichten[j],:,:]<thresholds[i][1],arr[schichten[j],:,:]>thresholds[i][0]),0,1), cmap ='gray')
    plt.show()

def patch_maker(directory, size):
    '''makes patches out of a dicom directory'''
    
    window_shape = (1, size[0], size[1])
    arr = dicom_to_array(directory)
    patch = view_as_windows(arr, window_shape)
    print(patch.shape)
 
def patch_label_random (array_image, array_label):
    '''generates patches from image and label arrays'''
    
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

multiple_nifti_arr([r'/home/eser/Downloads/Task05_Prostate/imagesTr/prostate_00.nii.gz',
                    r'/home/eser/Downloads/Task05_Prostate/imagesTr/prostate_01.nii.gz'])
print(list_arr)