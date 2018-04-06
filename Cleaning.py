# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:06:09 2018

@author: pulkit
"""

import Preprocess as prep


def clean_more(img,objects):
    maximum=0
    for i in objects:
        row_start,col_start,row_end,col_end=i
        image=img[row_start:row_end,col_start:col_end]
        image=prep.preprocess(image)
        height=image.shape[0]
        width=image.shape[1]
        change=0
        for row in range(height-1):
            for col in range(width-1):
                if (image[row][col] == 0 and image[row][col+1] == 255) or (image[row][col] == 255 and image[row][col+1] == 0):
                    change = change+1
        if(change>maximum):
            maximum = change
            plate = (row_start,col_start,row_end,col_end)
    return plate