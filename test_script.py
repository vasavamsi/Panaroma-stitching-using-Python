# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:48:49 2020

@author: ADMIN-PC
"""
from panaroma_stitching import panaroma

# Taking the input from user
img_set = int(input('enter the image set no. for panaroma stitching (enter from no. 1 to 5): ')) #change the as per given sets of images
path = './img_set_' + str(img_set) + '/'

#Calling the function
panaroma(path)
