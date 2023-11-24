# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:07:01 2020

@author: ADMIN-PC
"""
import cv2 as cv
import numpy as np
from ransac import RANSAC

def panaroma(path):

    stitch_cnt = 0
    fin_img = None
    for image in range(8,1,-1): ## Range can be changed in case the no. of images is different from 8.
        ## Taking the first image
        file_name = path + str(image) + '.jpg'
        img = cv.imread(file_name)
        img_1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_1 = cv.resize(img, None, fx=0.25, fy=0.25)
        
        ##Taking in the second image
        file_name = path + str(image-1) + '.jpg'
        img = cv.imread(file_name)
        img_2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
        img_2 = cv.resize(img, None, fx=0.25, fy=0.25)
        
        ## Appltying RANSAC Algorithm and obtaining homgraphy matrix
        Hp = RANSAC(img_1,img_2)
        if stitch_cnt >= 1:
            img_1 = cv.cvtColor(fin_img, cv.COLOR_BGR2RGB)
#            img_1 = fin_img
        
        ## Image stitching
        canvas = np.zeros(shape = [img_1.shape[0]*2, img_1.shape[1]*2, 3], dtype = np.uint8)
        
        ## Shifting the second image using homography estimation
        X = np.array([[0],[0],[1]])
        X_ = np.dot(Hp,X)

        canvas[0:img_1.shape[0], int(X_[0]):int(X_[0])+img_1.shape[1],:]  = img_1[:,:,:]  
#        for i in range(img_1.shape[0]):
#            for j in range(img_1.shape[1]):
#                X = np.array([[j], [i], [1]])
#                X_ = np.dot(Hp,X)
#                if X_[0] >= canvas.shape[1] or X_[1] >= canvas.shape[0]:
#                    continue
#                elif X_[0] < 0 or X_[1] < 0:
#                    continue
#                else:
#                    canvas[int(X_[1]),int(X_[0]),:] = img_1[i,j,:]
                
        ## Stitching the first image
        for i in range(img_2.shape[0]):
            for j in range(img_2.shape[1]):
                if canvas[i,j,0] == 0 and canvas[i,j,1] == 0 and canvas[i,j,2] == 0:
                    canvas[i,j,:] = img_2[i,j,:]
    
        ## Removing the blank space
        for j in range(canvas.shape[1]-1,0,-1):
            if canvas[200,j,0] != 0:
                break
        for i in range(canvas.shape[0]-1,0,-1):
            if canvas[i, 200, 0] != 0:
                break
            
        canvas = canvas[:i,:j,:]
        #######################################################################################################
#        fin_img = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)      ## UNCOMMENT TO SEE THE INTERMEDIATE PROGRESS
#        cv.imwrite(path + 'intermediate_panaroma.png', fin_img)
        #######################################################################################################
        stitch_cnt += 1 #increamenting the count
        
    ## Saving the final image
    fin_img = cv.cvtColor(fin_img, cv.COLOR_BGR2RGB)
    fin_img = cv.resize(fin_img, None, fx=1.5, fy=1.5)   
    cv.imwrite(path + 'final_panaroma.png', fin_img)
    
    print('the final stitched image is saved in the following path' + path)