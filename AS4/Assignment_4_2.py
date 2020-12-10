import cv2
import numpy as np
import sys
import math
import random

def main():

    op, ip, prob, nmin, nmax, kmax= readTxtFile()
    M,b, A = projectionmatM(op, ip)
    points3D, points2d = op, ip

    calculateParameters(M, b, points3D, points2d)
    op, ip, prob, nmin, nmax, kmax = readTxtFile()
    inlinerNum, bestM = ransac(op, ip, prob, nmin, nmax, kmax)
    print(inlinerNum, bestM)
    calculateParameters(bestM,b, op,ip)


def projectionmatM(points3D, points2D):
    A = np.zeros((len(points3D) * 2, 12))
    j = 0

    for i in range(0, len(points3D)):
        x = np.array([[points3D[i][0], points3D[i][1], points3D[i][2], 1, 0, 0, 0, 0,
                       -points2D[i][0] * points3D[i][0], -points2D[i][0] * points3D[i][1],
                       -points2D[i][0] * points3D[i][2], -points2D[i][0] * 1]])
        y = np.array([[0, 0, 0, 0, points3D[i][0], points3D[i][1], points3D[i][2], 1,
                       -points2D[i][1] * points3D[i][0], -points2D[i][1] * points3D[i][1],
                       -points2D[i][1] * points3D[i][2], -points2D[i][1] * 1]])
        A[j] = x
        A[j + 1] = y
        j += 2

    A = np.array(A)

    U, S, V = np.linalg.svd(A, full_matrices=True)
    V_transpose = V.T
    M = []
    row = 0
    b = []
    for i in range(0, 3):
        Projection_Matrix = []
        for j in range(0, 4):
            Projection_Matrix.append(V_transpose[row][11])
            if j == 3:
                b.append(V_transpose[row][11])
            row += 1

        M.append(Projection_Matrix)
    return M, b, A

def calculateParameters(M, b, points3D, points2D):
    np.set_printoptions(formatter={'float': "{0:.6f}".format})
    M = np.array(M)
    M.reshape(3, 4)
    m1 = np.array(M[0])
    m2 = np.array(M[1])
    m3 = np.array(M[2])
    actualPoint2DList = []

    # Finding Actual Points and Mean Square Error
    count = 0
    mse = 0
    points3 = points3D
    for points in points3:
        actualPoint2D = []
        points.append(1)
        points = np.array(points)

        xpoint2D = float(np.dot((m1.T), points)) / float(np.dot((m3.T), points))
        ypoint2D = float(np.dot((m2.T), points)) / float(np.dot((m3.T), points))
        actualPoint2D.append((xpoint2D, ypoint2D))
        actualPoint2DList.append(actualPoint2D)

        mse += pow((points2D[count][0] - xpoint2D), 2) + pow((points2D[count][1] - ypoint2D), 2)
        count += 1

    mse = mse / len(points2D)

    a1 = (m1[:3]).T
    a2 = (m2[:3]).T
    a3 = (m3[:3]).T

    ro = 1 / np.linalg.norm(a3)

    u0 = ro ** 2 * (np.dot(a1, a3))
    print("u0 = ", u0)

    v0 = ro ** 2 * (np.dot(a2, a3))
    print("v0 = ", v0)

    alfav = math.sqrt(float(np.dot(pow(ro ,2), np.dot(a2,a2)) - v0*v0))
    alfav = round(alfav,2)
    print("alfav = ", alfav)

    s = (float(alfav* np.dot(np.cross(a1,a3), np.cross(a2,a3))))
    s = math.ceil(s)
    print("s = ",s)

    alfau = (float(pow(ro, 2)*np.dot(a2,a2) - pow(s,2) - pow(v0, 2)) ** 0.5)
    alfau = round(alfau,2)
    print("alfau = ",alfau)

    K = np.array([[alfau, s, u0], [0, alfav, v0], [0, 0, 1]]).reshape(3, 3)

    print("K* = ", K)

    eps = np.sign(b[2])

    inv = np.linalg.inv(K)
    T = eps * ro * np.dot(inv, b)

    r3 = eps * ro * a3
    r1 = np.cross((alfav * a2), a3)
    r2 = np.cross(r1, r3)
    R = [r1, r2, r3]
    R = np.array(R).reshape(3, 3)

    r3 = eps * ro * a3
    r1 = ro ** 2 / alfav * np.cross(a2,a3)
    r2 = np.cross(r1, r3)
    Rstar = np.array([r1.T, r2.T, r3.T])

    print("T* = ", T)

    print("R* = ", Rstar)
    print("MSE =", mse)
    return M

def readTxtFile():
    filename = sys.argv[1]
    op,ip = [], []
    with open(filename) as f:
        data = f.readlines()
        for i in data:
            pt = i.split()
            op.append([float(p) for p in pt[:3]])
            ip.append([float(p) for p in pt[3:]])
    configname = sys.argv[2]
    with open(configname, 'r') as conf:
        prob = float(conf.readline().split()[0])
        kmax = int(conf.readline().split()[0])
        nmin = int(conf.readline().split()[0])
        nmax = int(conf.readline().split()[0])
    #return prob, nmin, nmax, kmax
    return op, ip, prob, nmin, nmax, kmax


def ransac(op, ip, prob, nmin, nmax, kmax):
    w = 0.5
    # k = math.log(1 - prob) / math.log(1 - (w ** nmin))
    k = kmax
    np.random.seed(0)
    count = 0
    inlinerNum = 0
    bestM = None
    m1,b1,a1 = projectionmatM(op,ip)
    fullD = distance(m1, op, ip)
    medianDistance = np.median(fullD)
    t = 1.5 * medianDistance
    n = random.randint(nmin, nmax)
    while (count < k and count < kmax):

        index = np.random.choice(len(op), n)
        ranOp, ranIp = np.array(op)[index], np.array(ip)[index]
        M,b,A = projectionmatM(ranOp, ranIp)
        d = distance(M, op, ip)
        inliner = []
        for i, d in enumerate(d):
            if d < t:
                inliner.append(i)
        if len(inliner) >= inlinerNum:
            inlinerNum = len(inliner)
            inlinerOp, inlinerIp = np.array(op)[inliner], np.array(ip)[inliner]
            M, b, A = projectionmatM(ranOp, ranIp)
            bestM = M
        if not (w == 0):
            w = float(len(inliner)) / float(len(ip))
            k = float(math.log(1 - prob)) / np.absolute(math.log(1 - (w ** n)))
        count += 1;
    return inlinerNum, bestM

def distance(M, op, ip):
    m1 = np.array(M[0])
    m2 = np.array(M[1])
    m3 = np.array(M[2])
    d = []
    for i, j in zip(op, ip):
        xi = j[0]
        yi = j[1]
        pi = np.array(i)
        # pi = np.concatenate([pi, [1]])
        pi = np.append(pi, 1)
        exi = (m1.T.dot( pi.T)) / (m3.T.dot(pi.T))
        eyi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        di = np.sqrt(((xi - exi) ** 2 + (yi - eyi) ** 2))
        d.append(di)
    return d

if __name__ == '__main__':
    main()