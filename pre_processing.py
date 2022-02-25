import cv2
import numpy as np
import os
from skimage.morphology import skeletonize as skl
from classifier import classification
import matplotlib.pyplot as plt
import math

ALPHA = 3
BETA = 0

def resizeAndPad(img, size, padColor=0):

	h, w = img.shape[:2]
	sh, sw = size

	# interpolation method
	if h > sh or w > sw: # shrinking image
		interp = cv2.INTER_AREA
	else: # stretching image
		interp = cv2.INTER_CUBIC

	# aspect ratio of image
	aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

	# compute scaling and pad sizing
	if aspect > 1: # horizontal image
		new_w = sw
		new_h = np.round(new_w/aspect).astype(int)
		pad_vert = (sh-new_h)/2
		pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
		pad_left, pad_right = 0, 0
	elif aspect < 1: # vertical image
		new_h = sh
		new_w = np.round(new_h*aspect).astype(int)
		pad_horz = (sw-new_w)/2
		pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
		pad_top, pad_bot = 0, 0
	else: # square image
		new_h, new_w = sh, sw
		pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

	# set pad color
	if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
		padColor = [padColor]*3

	# scale and pad
	scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
	scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

	return scaled_img

def processing_image(image = "test-images/rt.png"):
	# Input Image
	inputImage = cv2.imread(image)

	height, width = inputImage.shape[:2]
	max_height = 300
	max_width = 300

	# only shrink if img is bigger than required
	if max_height < height or max_width < width:
		# get scaling factor
		scaling_factor = max_height / float(height)
		if max_width/float(width) < scaling_factor:
			scaling_factor = max_width / float(width)
		# resize image
		inputImage = cv2.resize(inputImage, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

	# Copy for debugging purpose
	inputImageCopy = inputImage.copy()

	# Convert to grayscale
	grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

	# Threshold via Otsu algorithm:
	threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	cv2.floodFill(binaryImage, None, (0, 0), 0)
	
	# Find bounding boxes
	contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	boundingBoxes = [cv2.boundingRect(c) for c in contours]
	(cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
		key=lambda b:b[1][0], reverse=False))

	# --> meine creation <--

	mathSymbols = list()

	for c in boundingBoxes:
		x,y,w,h = c

		# Bounding box area
		rectArea = w * h

		# Minimum Area Threshold
		minArea = 25

		if rectArea > minArea:
			mathSymbols.append(c)
			
	mathSymbols = mathSymbols[6:]

	min_y_numbers = list()
	for i in range(len(mathSymbols)):
		min_y_numbers.append(mathSymbols[i][1])

	min_y = min(min_y_numbers)
	
	max_y_numbers = []
	for i in range(len(mathSymbols)):
		max_y_numbers.append(mathSymbols[i][1] + mathSymbols[i][3])

	max_y = max(max_y_numbers)

	#print(min_y)
	#print(max_y)
	
	exponents = list()
	for c in mathSymbols:
		#print(c[1]+c[3], " und ", ((max_y-min_y)*6/10)+min_y)
		if c[1]+c[3] <= math.ceil(((max_y-min_y)*6/10)+min_y):
			print(c, "Das ist ein Exponent!")
			exponents.append(c)
	
	#print(exponents)
	# --> meine creation <--

	count = 0
	for c in boundingBoxes:
		x,y,w,h = c

		# Bounding box area
		rectArea = w * h

		# Minimum Area Threshold
		minArea = 25

		if rectArea > minArea:
			cv2.rectangle(inputImageCopy,(x,y),(x+w,y+h),(0,255,0),2)
				
			# Crop bounding box:
			currentCrop = binaryImage[y:y+h,x:x+w]

			new_img = resizeAndPad(currentCrop, (45, 45))

			adjusted = cv2.convertScaleAbs(new_img, alpha=ALPHA, beta=BETA)
			reverse = 255-adjusted

			adjustedImageCopy = adjusted.copy()
			thinnedImage = adjustedImageCopy == 255
			thinnedImage = skl(thinnedImage)
			thinnedImage = thinnedImage.astype(np.uint8)*255

			reverse = 255-thinnedImage

			if c in exponents:
				print(c)
				cv2.imwrite("result-images/" + str(count) + "_exp" ".jpg", reverse)
				count += 1
				continue

			cv2.imwrite("result-images/" + str(count) + ".jpg", reverse)
			count += 1
	
	#plt.imshow(inputImageCopy)
	#plt.xticks([]), plt.yticks([])
	#plt.show()

if __name__ == "__main__":
	processing_image()
	