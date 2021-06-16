import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import math
from tqdm import *
import argparse
import os
import glob

def normalizePoints(pts):
    #calculate the mean of x and y coordinates - Compute the centroid of all corresponding points in a single image
    pts_mean = np.mean(pts, axis=0)
    x_bar = pts_mean[0]
    y_bar = pts_mean[1]

    #Recenter by subtracting the mean from original points
    x_tilda, y_tilda = pts[:,0] - x_bar, pts[:, 1] - y_bar

    #scale term s and s': average distances of the centered points from the origin in both the left and right images
    s = (2/np.mean(x_tilda**2 + y_tilda**2))**(0.5)

    #construct transformation matrix
    T_S = np.diag([s,s,1])
    T_T = np.array([[1, 0, -x_bar],[0, 1, -y_bar],[0, 0, 1]])
    T = np.dot(T_S, T_T)

    x = np.column_stack((pts, np.ones(len(pts))))
    x_norm = (T.dot(x.T)).T
    return x_norm, T

def getFundamentalMatrix(src, dst):
    col = 9
    # need minimum of 8 points to compute F Matrix
    if src.shape[0] > 7:
        #do normalization
        src_norm, T1 = normalizePoints(src)
        dst_norm, T2 = normalizePoints(dst)
        A = np.zeros((len(src_norm),9))

        for i in range(len(src_norm)):
            x1, y1 = src_norm[i][0], src_norm[i][1]
            x2, y2 = dst_norm[i][0], dst_norm[i][1]
            A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

        #calculate SVD of A
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        #F = VT.T[:, -1]
        #F = F.reshape(3,3)
        F = VT[-1,:]
        F = F.reshape(3,3)

        #the calculated F matrix can be of rank 3, but to find epipolar lines, the rank should be 2.
        #thus, 
        U_, S_, VT_ = np.linalg.svd(F)
        # make diagonal matrix and set last element to 0
        S_ = np.diag(S_)
        S_[2,2] = 0
        #recompute F
        F = np.dot(U_, np.dot(S_, VT_))

        #un-normalize
        F = np.dot(T2.T, np.dot(F, T1))
        return F
    else:
        return None

def calculateError(x1, x2, F):
    # make x1 and x2 3*3
    x1_ = np.array([x1[0], x1[1], 1])
    x2_ = np.array([x2[0], x2[1], 1])
    #calculate the error, the ideal case should be zero for the below product
    error = np.dot(x2_.T, np.dot(F, x1_))
    return np.abs(error)
