import cv2
import numpy as np
import sys
from PIL import Image
import os


# function to compute Intrinsic and Extrinsic Parameters
def computeIntrinsicExtrinsicParameters(a1, a2, a3, b):
    np.set_printoptions(formatter={'float': "{0:.6f}".format})  # set print options
    norm_P = 1 / np.linalg.norm(a3.T)  # |p| = 1 / |a3|
    u_0 = norm_P ** 2 * (a1.T.dot(a3))  # u0 = normP^2 * a1 * a3
    v_0 = norm_P ** 2 * (a2.T.dot(a3))  # v0 = normP^2 * a2 * a3
    a2dota2 = a2.T.dot(a2)  # a2dota2 = a2 dot a2
    a1crossa3 = np.cross(a1.T, a3.T)  # a1 X a3
    a2crossa3 = np.cross(a2.T, a3.T)  # a2 X a3
    a1dota1 = a1.T.dot(a1)  # a1 * a1

    alpha_v = np.sqrt(norm_P ** 2 * a2dota2 - v_0 ** 2)  # sqrt((p)^2 *a2 . a2 - v0^2)

    s = (norm_P ** 4) / alpha_v * a1crossa3.dot(a2crossa3.T)  # normP^4/ (alpha_V * (a1 X a3) . (a2 X a3) )

    alpha_u = np.sqrt(norm_P ** 2 * a1dota1 - s ** 2 - u_0 ** 2)  # sqrt((p)^2 *a1 . a1 - s^2 - u0^2)

    #	definning matrix K* = [alphaU 	s 		u0,
    #							0 		alphaV	v0,
    #							0		0		1]
    K_star = np.array([[alpha_u, s, u_0], [0, alpha_v, v_0], [0, 0, 1]])

    epsilon = np.sign(b[2])  # sigmoid(b3)
    T_star = epsilon * norm_P * np.linalg.inv(K_star).dot(b).T  # epsilon * |P| * K*^-1 * b
    r3 = epsilon * norm_P * a3  # epsilon * |P| * a3
    r1 = norm_P ** 2 / alpha_v * a2crossa3  # |P|^2 / alphaV a2 X a3
    r2 = np.cross(r3, r1)  # r3 X r1
    R_star = np.array([r1.T, r2.T, r3.T])  # [r1^T r2^T r3^T]^T
    print("--------------------------------------\n")
    print("v0 = %f \n" % u_0)
    print("u0 = %f \n" % v_0)
    print("alphaU = %f \n" % alpha_u)
    print("alphaV = %f \n" % alpha_v)
    print("s = %f \n" % s)
    print("K* = %s \n" % K_star)
    print("T* = %s \n" % T_star)
    print("R* = %s \n" % R_star)


# function to compute mean square error
def computeMeanSquareError(M, object_points, image_points):
    # defining m1,m2 and m3
    m1 = M[0][:4]
    m2 = M[1][:4]
    m3 = M[2][:4]
    mse = 0

    # mean square error = sum_1_to_N [(x_i - (m1^T*p_i/m3^T*p_i))^2 + (y_i - (m2^T*p_i/m3^T*p_i))^2] / n
    # calculating mean square error using above mentioned equation
    for i, j in zip(object_points, image_points):
        xi = j[0]
        yi = j[1]
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        exi = (m1.T.dot(pi)) / (m3.T.dot(pi))
        eyi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        mse += ((xi - exi) ** 2 + (yi - eyi) ** 2)
    mse = mse / len(object_points)
    print("--------------------------------------\n")
    print("Mean Square Error = %s\n" % mse)


# function to compute the value of matrix A
def computeMatrixA(object_points, image_points):
    A = []  # intializing matrix A
    zero = np.zeros(4)
    # calculating value of A line by line and Appending it to matrix
    for op, ip in zip(object_points,
                      image_points):  # zip() function returns an iterator of tuples based on the iterable object
        pi = np.array(op)
        pi = np.concatenate([pi, [1]])
        xipi = ip[0] * pi
        yipi = ip[1] * pi
        A.append(np.concatenate([pi, zero, -xipi]))
        A.append(np.concatenate([zero, pi, -yipi]))
    return np.array(A)  # returns matrix A


# function to compute matrix M and returning the value of a1,a2,a3 b and M
def computeMatrixM(A):
    M = []  # intializing matrix M
    u, s, v = np.linalg.svd(A,
                            full_matrices=True)  # getting value of u,s,v using single value decomposition of matrix A
    print("u = ",u)
    print("s = ", s)
    print("v = ", v)
    M = v[-1].reshape(3, 4)  # defining M from V
    print("M = ",M)
    # getting a1,a2,a3 from matrix M
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    b = []  # intializing matrix b
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1))  # reshaping matrix b
    return a1, a2, a3, b, M  # return a1,a2,a3,b,M


print(__doc__)

object_points, image_points = [], []  # intializing array for object and image points
data = open(
    'correspondencePoints.txt').readlines()  # reading data from file

# appending data to array of object and image points
for value in data:
    point = value.split()
    object_points.append([float(p) for p in point[:3]])
    image_points.append([float(p) for p in point[3:]])

A = computeMatrixA(object_points, image_points)  # computing matrix A from object point and image point
a1, a2, a3, b, M = computeMatrixM(A)  # computing matrix M and returning value of a1,a2,a3,b and M
computeIntrinsicExtrinsicParameters(a1, a2, a3, b)  # computing intrinsic and extrinsic parameters
computeMeanSquareError(M, object_points, image_points)  # computing mean square error