# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:17:45 2020

@author: islere
"""
import numpy as np
import dicom_lesen as dl
import matplotlib.pyplot as plt
min_max =[]
min_max_extended = []
def find_min_max(array):
    for i in array:
        if i != 0:
           min_max_extended.append(array.index(i)) 
    min_max.append(min_max_extended[0]) 
    min_max.append(min_max_extended[-1])
array = np.zeros([4,4,4])
array[2,2,2]=1
array[1,2,2]=1
array[0,1,1]=1
#find_min_max(array)
#print(min_max)

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

#crop_image(array)

def crop_image_only_outside(img,tol=0):
    # img is 3D image data
    # tol  is tolerance
    mask = img>tol
    
    m,n,o = img.shape
    masko,maskm,maskn = mask.any((0,1)),mask.any((1,2)),mask.any((0,2))
    print(masko)
    
    col_start,col_end = maskn.argmax(),n-maskn[::-1].argmax()
    
    row_start,row_end = maskm.argmax(),m-maskm[::-1].argmax()
    z_start, z_end = masko.argmax(),o-masko[::-1].argmax()
    return img[row_start:row_end,col_start:col_end, z_start:z_end]

image = dl.nifti_to_array(r'C:\Users\islere\Task01_BrainTumour\imagesTr\BRATS_002.nii.gz')

image = image[0]

plt.imshow(crop_image_only_outside(image)[90,:,:])
plt.show()
print(crop_image_only_outside(array))