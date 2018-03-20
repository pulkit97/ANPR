# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 01:52:51 2018

@author: pulkit
"""
from skimage import measure



def CCA_plate(binary_img):
    labels=measure.label(binary_img)
    plate_dims=(0.01*labels.shape[0],0.30*labels.shape[0],0.01*labels.shape[1],0.3*labels.shape[1])
    obj_coord=[]
    min_height, max_height, min_width, max_width = plate_dims
    #Processing each connnected component and looking for number plate like objects and constructung borders for illustration
    for region in measure.regionprops(labels):
        if region.area<50:
            continue
        row_start,col_start,row_end,col_end=region.bbox
        height=row_end-row_start
        width=col_end-col_start
        if(height>=min_height and width>=min_width and height<=max_height and width<=max_width and height<width):
            obj_coord.append((row_start,col_start,row_end,col_end))
    return obj_coord

