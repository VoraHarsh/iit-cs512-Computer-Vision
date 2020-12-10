import cv2
import numpy as np
import sys

def getImage():
    if len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])
    else:
        print("Please input an Image as an Argument.")
    return image

def main():

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    coords = np.zeros((6*7,3), np.float32)
    coords[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    points3D = []
    points2D = []

    image = getImage()

    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(image_gray,(7,6),None)

    if ret == True:
        points3D.append(coords)

        corners_new = cv2.cornerSubPix(image_gray,corners,(11,11),(-1,-1),criteria)
        points2D.append(corners_new)

        cv2.drawChessboardCorners(image, (7,6), corners_new,ret)
        cv2.imshow('image',image)

    file = open("correspondencePoints.txt", "w")
    for i, j in zip(coords, corners.reshape(-1, 2)):
        file.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(j[0]) + ' ' + str(j[1]) + '\n')
    file.close()

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()