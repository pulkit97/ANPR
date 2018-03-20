# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:49:50 2018

@author: pulkit
"""

#Importing the required libraries
from skimage.io import imread
import matplotlib.pyplot as plt
import Preprocess as prep
import CCA_plate as ccp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import CCA_Segmentation as ccs
from keras.models import load_model
import numpy as np
import Cleaning
#Reading the data as Greyscale
img=imread("ImageData\Demo_b.jpg",as_grey=True)
img_obj=[]
#Displaying the image
fig,(ax1)=plt.subplots(1)
ax1.imshow(img,cmap="gray")
fig.suptitle('Input Image', fontsize=14, fontweight='bold')

#Preprocessing the image
preprocessed_img=prep.preprocess(img)
fig,(ax1)=plt.subplots(1)
ax1.imshow(preprocessed_img,cmap="gray")
fig.suptitle('Preprocessed Image', fontsize=14, fontweight='bold')

#Applying CCA #Returns coordinates of plate-like-objects
plate_like_objects=ccp.CCA_plate(preprocessed_img)


#Drawing Bounding Boxes around the plate like objects
fig,(ax1)=plt.subplots(1)
ax1.imshow(img,cmap="gray")
for i in plate_like_objects:
    row_start,col_start,row_end,col_end=i
    rectBorder = patches.Rectangle((col_start, row_start), col_end-col_start, row_end-row_start, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)
    img_obj.append(img[row_start:row_end,col_start:col_end])
fig.suptitle('Localised Objects Using CCA', fontsize=14, fontweight='bold')

#Yet to apply filtering methods so hard coded

plate_coord=Cleaning.clean_more(img,plate_like_objects)
row_start,col_start,row_end,col_end=plate_coord
plate=img[row_start:row_end,col_start:col_end]


fig,(ax1)=plt.subplots(1)
ax1.imshow(plate,cmap="gray")
fig.suptitle('Number Plate', fontsize=14, fontweight='bold')

#Preprocessing the localised number plate
threshold_val=threshold_otsu(plate)
binary_plate=plate>threshold_val

fig,(ax1)=plt.subplots(1)
ax1.imshow(binary_plate,cmap="gray")
fig.suptitle('Preprocessed Plate Image', fontsize=14, fontweight='bold')

#Performing CCA again on the localised plate
characters,char_order=ccs.CCA_segmentation(binary_plate)


'''fig,(ax1)=plt.subplots(1)
ax1.imshow(characters[1],cmap="gray")'''


#Prediction

model = load_model('Neural_Net_Models/CNN.h5')
characters=np.reshape(characters,(10,28,28,1))



asci=47
diction={1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'A', 12: 'B', 13: 'C',
 14: 'D',15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R',
 29: 'S',30: 'T',31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y',
 36: 'Z'}

out=model.predict_classes(characters)
out_str=""
for i in out:
    out_str=out_str+diction[i]


#Ordering the string output
column_list_copy = char_order[:]    
char_order.sort()
rightplate_string=''
for each in char_order:
    rightplate_string += out_str[column_list_copy.index(each)]
print(rightplate_string)

        
