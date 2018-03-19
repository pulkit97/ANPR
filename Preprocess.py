# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:49:12 2018

@author: pulkit
"""
#Importing the required libraries
from skimage.filters import threshold_otsu
from skimage.filters import median
from skimage.filters import sobel
from skimage.morphology import disk
#Function to preprocess

def preprocess(img):
    
    #Applying otsu method for binarization
    threshold_val=threshold_otsu(img)
    binary_img=img>threshold_val
    med=median(binary_img,disk(5))
    return med
    
