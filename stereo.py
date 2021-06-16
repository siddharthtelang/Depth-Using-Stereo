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

# https://cmsc733.github.io/2019/proj/p3/
def processInliers(src_pts, dst_pts):
    max_error = 0.001
    iterations = 1
    final_idx = []
    F_Matrix = None
    inliers = 0
    rows = src_pts.shape[0]
    for i in range(2000):
        temp_idx = []
        random_row = np.random.choice(rows, size=8)
        src = src_pts[random_row]
        dst = dst_pts[random_row]
        f_matrix = getFundamentalMatrix(src, dst)
        #now check the F matrix for all pairs
        for j in range(rows):
            error = calculateError(src_pts[j], dst_pts[j], f_matrix)
            if error < max_error:
                temp_idx.append(j)
        
        if len(temp_idx) > inliers:
            inliers = len(temp_idx)
            final_idx = temp_idx
            F_Matrix = f_matrix
            src_final = src_pts[final_idx]
            dst_final = dst_pts[final_idx]
    
    return F_Matrix, src_final, dst_final

def getEssentialMatrix(F, K1, K2):
    E = K2.T.dot(F).dot(K1)
    #enforce rank 2 for singular matrix
    U,S,VT = np.linalg.svd(E)
    S = np.diag([1,1,0])
    E = np.dot(U, np.dot(S, VT))
    return E

#https://cmsc733.github.io/2019/proj/p3/
def getCameraPose(E):
    U, S, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R, T = [],[]
    R.append(np.dot(U, np.dot(W, VT)))
    R.append(np.dot(U, np.dot(W, VT)))
    R.append(np.dot(U, np.dot(W.T, VT)))
    R.append(np.dot(U, np.dot(W.T, VT)))
    T.append(U[:, 2])
    T.append(-U[:, 2])
    T.append(U[:, 2])
    T.append(-U[:, 2])

    # R should always be positive
    for i in range(len(R)):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            T[i] = -T[i]
    
    return R, T

def getCorrectPose(pts_3D, R1, T1, R2, T2):
    num_Z_positive = 0
    zList = []
    I = np.identity(3)
    for k in range (len(pts_3D)):
        num_Z_positive = 0
        pts3D = pts_3D[k]
        #normalize
        pts3D = pts3D/pts3D[3, :]

        P_2 = np.dot(R2[k], np.hstack((I, -T2[k].reshape(3,1))))
        P_2 = np.vstack((P_2, np.array([0,0,0,1]).reshape(1,4)))

        P_1 = np.dot(R1, np.hstack((I, -T1.reshape(3,1))))
        P_1 = np.vstack((P_1, np.array([0,0,0,1]).reshape(1,4)))

        for i in range(pts3D.shape[1]):
            #calculate point for Right image
            X_2 = pts3D[:,i]
            X_2 = X_2.reshape(4,1)
            Xc_2 = np.dot(P_2, X_2)
            Xc_2 = Xc_2 / Xc_2[3]
            z_2 = Xc_2[2]

            #calcuate points for Left image
            X_1 = pts3D[:,i]
            X_1 = X_1.reshape(4,1)
            Xc_1 = np.dot(P_1, X_1)
            Xc_1 = Xc_1 / Xc_1[3]
            z_1 = Xc_1[2]

            if (z_1 > 0 and z_2 > 0):
                num_Z_positive += 1

        #print(num_Z_positive)
        zList.append(num_Z_positive)

        # get the correct camera pose index - define threshold for points as half the number of points
        threshold = pts_3D[0].shape[1]//2
        zArray = np.array(zList)
        index = np.where(zArray > threshold)

    return index
