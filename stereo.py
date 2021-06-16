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
