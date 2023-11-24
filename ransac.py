# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:50:34 2020

@author: ADMIN-PC
"""
import cv2 as cv
import pandas as pd
import numpy as np
from numpy.linalg import norm
from homography import homograpy_matrix
import random

def RANSAC(img_1, img_2):
    """
    Arguments:
        
    img_1 = the first image from which the homography matrix to be calculated
    img_2 = the second image from which the homographhy matrix to be calculated
    
    Returns:
        
    F = Fundamental Matrix after going through RANSAC Algorithm
    
    """
    ##Taking care of the detector, i.e SIFT detector
    det = cv.xfeatures2d.SIFT_create()
    
    #Getting keypoints and their feature descriptor for first image
    kp1, des1 = det.detectAndCompute(img_1, None)
    des1 = np.array(des1, dtype = "float32")

    #Getting keypoints and their feature descriptor for first image
    kp2, des2 = det.detectAndCompute(img_2, None)
    des2 = np.array(des2)
    
    # key point matching
    coord1 = []
    coord2 = []
    dist_list = []
    for i in range(des1.shape[0]):
        diff_mat = des1[i,:] - des2
        dist =  norm(diff_mat, axis = 1)
        k2 = np.argmin(dist)
        coord1.append(kp1[i].pt)
        coord2.append(kp2[k2].pt)
        dist_list.append(np.min(dist))
    
    dist_df = pd.DataFrame(dist_list)
    dist_df = dist_df[0].sort_values()
    
    ##RANSAC Algorithm
    N = 10000    ## Taken tentatively and can be changed as per need
    Si_list = []
    n_inliers_list = []
    for iteration in range(0,N):
        # randomly sampling the 4 key points and determining the homography matrix
        rand_samples = random.sample(dist_df.index[:50], 4)
        
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for i in rand_samples:
            x1.append(coord1[i][0])
            x2.append(coord2[i][0])
            y1.append(coord1[i][1])
            y2.append(coord2[i][1])
        
        # Getting Homography Matrix for the Random Sample
        Hp = homograpy_matrix(x1,x2,y1,y2,img_1.shape[0],img_1.shape[1],img_2.shape[0],img_2.shape[1])
        
        # Getting the no. of inliers
        n_inliers = 0
        Si = []
        for n in dist_df.index[:50]:
            X = np.array([[coord1[n][0]], [coord1[n][1]], [1]])
            X_ = np.array([[coord2[n][0]], [coord2[n][1]], [1]])
            criteria = np.linalg.norm(X_ - np.dot(Hp,X))
            if abs(criteria) < 5:
                Si.append(n)
                n_inliers += 1
        Si_list.append(Si)
        n_inliers_list.append(n_inliers)
        
        if n_inliers >= 15:   ## Here the treshold for the inliers can be changed as per user requirement
            break
                
    print("RANSAC Algorithm stopped at " + str(iteration))
    print("Maximum no. of inliers obtained are " + str(max(n_inliers_list)))
    
    #Getting the final Homography matrix    
    index = np.argmax(np.array(n_inliers_list))
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in Si_list[index]:
        x1.append(coord1[i][0])
        x2.append(coord2[i][0])
        y1.append(coord1[i][1])
        y2.append(coord2[i][1])
    Hp = homograpy_matrix(x1,x2,y1,y2,img_1.shape[0],img_1.shape[1],img_2.shape[0],img_2.shape[1])
    print('The Homography matrix for the current pair of images is as follows')
    print(Hp)
    return Hp
