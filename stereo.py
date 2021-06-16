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

def getEpipolarLines(src_final, dst_final, F, im1_epipolar, im2_epipolar, rectified=False):
    lines1, lines2 = [], []
    for i in range(len(src_final)):
        #arrange the source and destination points in a 3*3 array to be multiplied with F
        x1 = np.array([src_final[i,0], src_final[i,1], 1]).reshape(3,1)
        x2 = np.array([dst_final[i,0], dst_final[i,1], 1]).reshape(3,1)

        #epipolar line 1 coefficients - left image
        line1 = np.dot(F.T, x2)
        lines1.append(line1)

        #epipolar line 2 coefficients - right image
        line2 = np.dot(F, x1)
        lines2.append(line2)

        if (not rectified):
            #get the x and y values - lines are not parallel to x axis
            x1_low = 0
            x1_high = im1_epipolar.shape[1] - 1
            y1_low = -(line1[2] + x1_low*line1[0])/line1[1]
            y1_high = -(line1[2] + x1_high*line1[0])/line1[1]

            x2_low = 0
            x2_high = im2_epipolar.shape[1] - 1
            y2_low = -(line2[2] + x2_low*line2[0])/line2[1]
            y2_high = -(line2[2] + x2_high*line2[0])/line2[1]
        
        else:
            # as the lines are parallel to the X axis, the slope tends to zero
            x1_low = 0
            x1_high = im1_epipolar.shape[1] - 1
            y1_low = -(line1[2]/line1[1])
            y1_high = y1_low

            x2_low = 0
            x2_high = im2_epipolar.shape[1] - 1
            y2_low = -(line2[2]/line2[1])
            y2_high = y2_low

        #print the points onto image
        cv2.circle(im1_epipolar, (int(src_final[i,0]), int(src_final[i,1])), 5, (0,0,255), 2)
        im1_epipolar = cv2.line(im1_epipolar, (int(x1_low), int(y1_low)), (int(x1_high), int(y1_high)), (0,255,0), 1)

        cv2.circle(im2_epipolar, (int(dst_final[i,0]), int(dst_final[i,1])), 5, (0,0,255), 2)
        im2_epipolar = cv2.line(im2_epipolar, (int(x2_low), int(y2_low)), (int(x2_high), int(y2_high)), (0,255,0), 1)

    combined = np.concatenate((im1_epipolar, im2_epipolar), axis=1)
    # temp = cv2.resize(combined, (1200,700))
    # cv2.imshow('epilines', temp)
    # cv2.imwrite('Epilines_.png', combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lines1, lines2, combined


def SSD(mat1, mat2):
    diff_sq = np.square(mat1 - mat2)
    ssd = np.sum(diff_sq)
    return ssd

def SAD(mat1, mat2):
    return np.sum(abs(mat1 - mat1))

def calcuateDisparity(img1_rectified_reshaped, img2_rectified_reshaped):
    h, w = img1_rectified_reshaped.shape
    disparity = np.zeros((h, w), np.uint8)
    window_size = 11
    half_window_size = math.floor((window_size)/2)
    search_distance = 200

    for row in tqdm(range(half_window_size, h -  half_window_size)): 
        for col in range(half_window_size, w -  half_window_size):
            patch1 = img1_rectified_reshaped[row - half_window_size: row + half_window_size, col - half_window_size : col + half_window_size]

            min_ssd = 10000
            disp = 0
            #scan along epiline till a particular length
            for distance in range(-search_distance, search_distance, 1): #bidirectional
                c_dash = col + distance
                # print(c_dash)
                if (c_dash < w - half_window_size) and (c_dash > half_window_size):
                    patch2 = img2_rectified_reshaped[row - half_window_size: row + half_window_size, c_dash - half_window_size : c_dash + half_window_size]
                    # if patch2.shape[1] < 4:
                    #     print(r, c, c_dash)
                    ssd = SSD(patch1, patch2)
                    if ssd < min_ssd:
                        min_ssd = ssd
                        disp = np.abs(distance)

            disparity[row, col] = disp

    return disparity



def Stereo(args):
    dataset = args['set']
    if (dataset == 1):
        K1 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
        K2 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
        baseline = 177.288
        f = K1[0,0]
    elif (dataset == 2):
        K1 = np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]])
        K2 = np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])
        baseline = 144.049
        f = K1[0,0]
    elif (dataset == 3):
        K1 = np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]])
        K2 = np.array([[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]])
        baseline = 174.019
        f = K1[0,0]
    else:
        print('Invalid Data Set...Exit')
        return

    folder = args['full_path']
    if (not folder[-1] == '/'):
        folder = folder + '/'
    img1 = cv2.imread(folder+'im0.png')
    img2 = cv2.imread(folder+'im1.png')
    im1 = img1.copy()
    im2 = img2.copy()
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    #create SIFT object and get the points
    print('Matching using SIFT in process')
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray,None)
    kp2, des2 = sift.detectAndCompute(img2_gray,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
    #select only 100
    #good_matches = good_matches[:100]
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,flags=2)
    #plt.imshow(img3)
    #plt.show()
    #cv2.imwrite('matches.png', img3)

    # get the points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

    # get the F matrix using RANSAC and the inlier points
    print('Fundamental Matrix is being calculated.....')
    F, src_final, dst_final = processInliers(src_pts, dst_pts)
    print('Fundamental Matrix using RANSAC estimation = ')
    print(F)
    print('')

    # combine the two images side by side and draw lines
    comb = np.concatenate((im1, im2), axis=1)
    s1 = src_final[:,0].astype(int)
    s2 = src_final[:,1].astype(int)
    d1 = dst_final[:,0].astype(int)
    d2 = dst_final[:,1].astype(int)
    d1 += im1.shape[1]
    for i in range(s1.shape[0]):
        cv2.line(comb, (s1[i], s2[i]), (d1[i], d2[i]), (0,0,255), 2)
    #cv2.imwrite('combined.png', comb)
    #plt.imshow(comb)

    # get the essential 
    print('Essential Matrix is being calculated.....')
    E = getEssentialMatrix(F, K1, K2)
    print('Essential Matrix = ')
    print(E)
    print('')

    #get the Camera pose
    R1 = np.identity(3)
    T1 = np.zeros((3,1))
    print('Camera pose estimation in process')
    R2, T2 = getCameraPose(E)

    #estimate the 3D points
    pts_3D = []
    #reference Rotation, Translation and Projection matrix-
    R1 = np.identity(3)
    T1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -T1.reshape(3,1)))))

    for i in range(len(R2)):
        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -T2[i].reshape(3,1)))))
        X_3D = cv2.triangulatePoints(P1, P2, src_final.T, dst_final.T)
        pts_3D.append(X_3D)

    #get the correct R2 and T2 index out of 4 values 
    print('Getting the correct pose')
    correct_index = (getCorrectPose(pts_3D, R1, T1, R2, T2))[0][0]
    R2_final = R2[correct_index]
    T2_final = T2[correct_index]
    print('Correct R and T Matrix:')
    print(R2_final)
    print(T2_final)
    print('')

    # get the epipolar lins
    lines1, lines2, epipolarLinesCombined = getEpipolarLines(src_final, dst_final, F, im1.copy(), im2.copy())
    #plt.imshow(epipolarLinesCombined)
    #cv2.imwrite('Epipolar_combined.png',epipolarLinesCombined)

    # Rectification
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(src_final), np.float32(dst_final), F, imgSize=img1.shape[1::-1])
    img1_rectified = cv2.warpPerspective(im1, H1, img1.shape[1::-1])
    img2_rectified = cv2.warpPerspective(im2, H2, img1.shape[1::-1])
    combined = np.concatenate((img1_rectified, img2_rectified), axis=1)
    #cv2.imwrite('Rectified Combined.png', combined)
    #plt.imshow(combined)
    print('H1 = ',H1)
    print('H2 = ',H2)
    print('')

    # get the rectified points
    src_final_rectified = cv2.perspectiveTransform(src_final.reshape(-1, 1, 2), H1).reshape(-1,2)
    dst_final_rectified = cv2.perspectiveTransform(dst_final.reshape(-1, 1, 2), H2).reshape(-1,2)

    im1_rectified_with_pts = img1_rectified.copy()
    im2_rectified_with_pts = img2_rectified.copy()

    im1_rectified = img1_rectified.copy()
    im2_rectified = img2_rectified.copy()

    #draw the points onto image
    for i in range(src_final_rectified.shape[0]):
        cv2.circle(im1_rectified_with_pts, (int(src_final_rectified[i,0]), int(src_final_rectified[i,1])), 10, (0,0,255), 2)
        cv2.circle(im2_rectified_with_pts, (int(dst_final_rectified[i,0]), int(dst_final_rectified[i,1])), 10, (0,0,255), 2)
    # cv2.imwrite('Rectified_img1_with_Points.png', im1_rectified_with_pts)
    # cv2.imwrite('Rectified_img2_with_Points.png', im2_rectified_with_pts)
    # cv2.imwrite('Rectified_img1.png', im1_rectified)
    # cv2.imwrite('Rectified_img2.png', im2_rectified)

    # calculate the rectified F matrix
    H2_T_inv =  np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F, H1_inv))

    img1_rectified_epipolar = im1_rectified.copy()
    img2_rectified_epipolar = im2_rectified.copy()

    lines1_rectified, lines2_rectified, epipolarLinesCombined_rectified = getEpipolarLines(src_final_rectified, dst_final_rectified, F_rectified, img1_rectified_epipolar, img2_rectified_epipolar, True)
    #cv2.imwrite('img1_rectified_epipolar.png', img1_rectified_epipolar)
    #cv2.imwrite('img2_rectified_epipolar.png', img2_rectified_epipolar)
    #plt.imshow(epipolarLinesCombined_rectified)

    #convert to grayscale and reshape the image
    img1_rectified_reshaped = cv2.cvtColor(im1_rectified, cv2.COLOR_BGR2GRAY)
    img1_rectified_reshaped = cv2.resize(img1_rectified_reshaped, (600,400))
    img2_rectified_reshaped = cv2.cvtColor(im2_rectified, cv2.COLOR_BGR2GRAY)
    img2_rectified_reshaped = cv2.resize(img2_rectified_reshaped, (600,400))

    # calculate the disparity
    disparity = calcuateDisparity(img1_rectified_reshaped, img2_rectified_reshaped)
    #scale it to max 255
    disparity_map = np.uint8(disparity * (255 / np.max(disparity)))
    plt.figure('Disparity Map')
    plt.imshow(disparity_map, cmap='hot',interpolation='nearest')
    plt.savefig('disparity.png')

    # calculate depth, limit it and rescale to 255
    depth = (baseline*f) / (disparity_map + 1e-15)
    print(np.max(depth))
    depth[depth > 100000] = 100000
    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.figure('Depth Map')
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('depth_image.png')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--set", required=False, help="dataset1/2/3", default='3', type=int)
    parser.add_argument("-path", "--full_path", required=False, help="working directory of the dataset", default='DataSets/Dataset 3/', type=str)
    args = vars(parser.parse_args())
    if (not os.path.exists(args['full_path'])):
        print('Path does not exist ; Re run and enter correct path as per README')
        exit()
    Stereo(args)