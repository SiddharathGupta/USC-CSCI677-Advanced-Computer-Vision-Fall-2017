"""
Watershed Segmentor
Homework-2 Part-b (Method-1)
based on concept reference: http://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1
Chinmay Chinara
USC ID: 2527237452
Date: 26th September 2017
"""

import numpy as np
import cv2

# Reading image
img = cv2.imread('101085.jpg', 3)
# img = cv2.imread('300091.jpg', 3)
# img = cv2.imread('253027.jpg', 3)

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Pyramid mean shifted filtering with spatial window radius 21 and color window radius 51.
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

# display the input image
cv2.imshow('Input_Image', img)
cv2.moveWindow('Input_Image', 0, 0)

# display the shifted image
cv2.imshow('Pyramid_Shifted_image', shifted)
cv2.moveWindow('Pyramid_Shifted_image', np.size(img, 1)+10, 0)

# Thresholding by otsu and binary inverse
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# watershed algorithm applied over the image
markers = cv2.watershed(img, markers)

# show the segmented lines over the image
img[markers == -1] = [255, 0, 0]
cv2.imshow('Output_segmented', img)
cv2.moveWindow('Output_segmented', 0, np.size(img, 0)+60)

for i in range(0,np.amax(markers)+1):
    img[markers == i] = [np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255)]

# fill different colors over the segmented regions for better clarity
cv2.imshow('Output_color_segmented', img)
cv2.moveWindow('Output_color_segmented', np.size(img, 1)+10, np.size(img, 0)+60)

# wait for ESC key to exit
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()