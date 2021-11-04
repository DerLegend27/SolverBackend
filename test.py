# Imports:
from PIL import Image
import pytesseract
import cv2
import os
import numpy as np

cropped_images = 0


# Read Input image
inputImage = cv2.imread("/Users/nicolai/Documents/imageprocessing/Images/example1.jpeg")

# Deep copy for results:
inputImageCopy = inputImage.copy()

# Convert BGR to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

cv2.imshow("test", grayscaleImage)
cv2.waitKey(0)

# Threshold via Otsu:
#threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 85, 10)
cv2.imshow("test", binaryImage)
cv2.waitKey(0)

# Flood-fill border, seed at (0,0) and use black (0) color:
cv2.floodFill(binaryImage, None, (0, 0), 0)


cv2.imshow("test", binaryImage)
cv2.waitKey(0)

# Get each bounding box
# Find the big contours/blobs on the filtered image:
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Look for the outer bounding boxes (no children):
for _, c in enumerate(contours):

    # Get the bounding rectangle of the current contour:
    boundRect = cv2.boundingRect(c)

    # Get the bounding rectangle data:
    rectX = boundRect[0]
    rectY = boundRect[1]
    rectWidth = boundRect[2]
    rectHeight = boundRect[3]

    # Estimate the bounding rect area:
    rectArea = rectWidth * rectHeight

    # Set a min area threshold
    minArea = 15

    # Filter blobs by area:
    if rectArea > minArea:

        cropped_images += 1

        # Draw bounding box:
        color = (0, 255, 0)
        cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                      (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
        #cv2.imshow("Bounding Boxes", inputImageCopy)

        # Crop bounding box:
        currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
        #cv2.imshow("Current Crop", currentCrop)
        #cv2.waitKey(0)
        cv2.imwrite("cropped_" + str(cropped_images) + ".png", currentCrop)

cv2.imshow("Boxes", inputImageCopy)
cv2.waitKey(0)