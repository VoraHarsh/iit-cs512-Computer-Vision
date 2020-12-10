import cv2
import numpy as np
import sys

def main():
	combine, image1, image2 = recieveImage()
	print("input key to Process image(press 'H' for help, press 'q' to quit):")
	key = input()
	while key != 'q':
		if key == 'c':
			n = input("Enter the varience of Gaussian scale(n):")
			wsize = input("Size of the Window :")
			k = input("Enter the weight of the trace in the harris corner detector between (0, 0.5)(k):")
			threshold = input("Enter Threshold Value:")
			print("Processing the images...")
			result = corner_det(combine, n, wsize, k, threshold)
			cv2.imwrite("cornerdet1.jpg", result)
			showWindow(result)
		if key == 'f':
			result = featureVector(image1, image2)
			cv2.imwrite("featurevector1.jpg", result)
			showWindow(result)
		if key == 'b':
			image = combine
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			gray = np.float32(gray)
			dst = cv2.cornerHarris(gray, 2, 3, 0.04)
			dst = cv2.dilate(dst, None)
			ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
			dst = np.uint8(dst)
			ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
			corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
			result = np.hstack((centroids, corners))
			result = np.int0(result)
			image[result[:, 1], result[:, 0]] = [0, 0, 255]
			image[result[:, 3], result[:, 2]] = [0, 255, 0]
			result = image
			cv2.imwrite("localization1.jpg", result)
			showWindow(result)
		if key == 'H':
			print("'c': Estimate image gradients and apply Harris corner detection algorithm.")
			print("'b': Obtain a better localization of each corner.")
			print("'f': Compute feature vectors for each corner that were detected.\n")
		print("input key to Process image(press 'H' for help, press 'q' to quit):")
		key = input()


def recieveImage():
	if len(sys.argv) == 3:
		image1 = cv2.imread(sys.argv[1])
		image2 = cv2.imread(sys.argv[2])
	else:
			cap = cv2.VideoCapture(0)
			for i in range(0,15):
				retval1,image1 = cap.read()
				retval2,image2 = cap.read()
			if retval1 and retval2:
				cv2.imwrite("capture12.jpg", image1)
				cv2.imwrite("capture23.jpg", image2)

	image1 = to3channel(image1)
	image2 = to3channel(image2)

	while image1.shape[0] > 1200 or image1.shape[1] > 750:
		image1 = cv2.resize(image1, (int(image1.shape[1] / 2), int(image1.shape[0] / 2)))

	while image2.shape[0] > 1200 or image2.shape[1] > 750:
		image2 = cv2.resize(image2, (int(image2.shape[1] / 2), int(image2.shape[0] / 2)))

	combine = np.concatenate((image1, image2), axis=1)
	cv2.imwrite("combined.jpg", combine)
	return combine, image1, image2;


def to3channel(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.GRAY2BGR)
    return image

def showWindow(image):
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def corner_det(image, n, wsize, k, threshold):
	n = int(n)
	wsize = int(wsize)
	k = float(k)
	threshold = int(threshold)
	copy = image.copy()
	rList = []
	height = image.shape[0]
	width = image.shape[1]
	offset = int(wsize / 2)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.float32(image)
	kernel = np.ones((n, n), np.float32) / (n * n)
	image = cv2.filter2D(image, -1, kernel)
	dy, dx = np.gradient(image)
	Ixx = dx ** 2
	Ixy = dy * dx
	Iyy = dy ** 2

	for y in range(offset, height - offset):
			for x in range(offset, width - offset):
				windowIxx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIxy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				Sxx = windowIxx.sum()
				Sxy = windowIxy.sum()
				Syy = windowIyy.sum()
				det = (Sxx * Syy) - (Sxy ** 2)
				trace = Sxx + Syy
				r = det - k *(trace ** 2)
				rList.append([x, y, r])
				if r > threshold:
							copy.itemset((y, x, 0), 0)
							copy.itemset((y, x, 1), 0)
							copy.itemset((y, x, 2), 255)
							cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
	return copy


def featureVector(image1, image2):
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(image1,None)
	kp2, des2 = orb.detectAndCompute(image2,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	kp1List = []
	kp2List = []
	for m in matches:
		(x1, y1) = kp1[m.queryIdx].pt
		(x2, y2) = kp2[m.trainIdx].pt
		kp1List.append((x1, y1))
		kp2List.append((x2, y2))
	for i in range(0, 50):
		point1 = kp1List[i]
		point2 = kp2List[i]
		cv2.putText(image1, str(i), (int(point1[0]), int(point1[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
		cv2.putText(image2, str(i), (int(point2[0]), int(point2[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	result = np.concatenate((image1, image2), axis=1)
	return result


if __name__ == '__main__':
	main()