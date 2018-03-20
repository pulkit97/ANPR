# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:28:49 2018

@author: pulkit
"""

import cv2
from skimage import measure
import numpy as np
from skimage.transform import resize

def CCA_segmentation(plate):
    orig_plate=plate
    plate=np.invert(plate)
    labels = measure.label(plate, neighbors=8, background=0)
    #Processing each label
    characters=[]
    column_list=[]
    for label in np.unique(labels):
        if label == -1:
            continue
        
        labelMask = np.zeros(plate.shape, dtype="uint8")
        labelMask[labels == label] = 255
        
        (_, cnts,_) = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate.shape[0])
            
            keepAspectRatio = aspectRatio < 1
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.20 and heightRatio < 0.60
            
            
            if keepAspectRatio and keepSolidity and keepHeight:
                img=resize(orig_plate[boxY-2:boxY+boxH+2,boxX-1:boxX+boxW+1],(28,28))
                characters.append(img)
                column_list.append(boxX)
                
    return (characters,column_list)
                

    
    