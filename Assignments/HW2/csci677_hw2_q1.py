"""
Mean Shift Segmentor
Homework-2 Part-a
Concept reference: http://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0
Chinmay Chinara
USC ID: 2527237452
Date: 26th September 2017
"""

import cv2
import numpy as np

# read the input image
# img = cv2.imread('101085.jpg',3)
# img = cv2.imread('300091.jpg',3)
img = cv2.imread('253027.jpg',3)

# Convert the input image to LAB space
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

# Level-1 pyramid with different spatial window radius and color window radius
out1 = cv2.pyrMeanShiftFiltering(lab, 85, 51)
out2 = cv2.pyrMeanShiftFiltering(lab, 21, 51)
out3 = cv2.pyrMeanShiftFiltering(lab, 30, 30)
out4 = cv2.pyrMeanShiftFiltering(lab, 12, 25)
out5 = cv2.pyrMeanShiftFiltering(lab, 200, 30)
out6 = cv2.pyrMeanShiftFiltering(lab, 11, 60)
out7 = cv2.pyrMeanShiftFiltering(lab, 30, 200)

# display the images
cv2.imshow('input', lab)
cv2.moveWindow('input', 0, 0)

cv2.imshow('output_85_51', out1)
cv2.moveWindow('output_85_51', np.size(img, 1)+10, 0)

cv2.imshow('output_21_51', out2)
cv2.moveWindow('output_21_51', (2*np.size(img, 1))+20, 0)

cv2.imshow('output_30_30', out3)
cv2.moveWindow('output_30_30', (3*np.size(img, 1))+30, 0)

cv2.imshow('output_12_25', out4)
cv2.moveWindow('output_12_25', 0, (np.size(img, 0))+60)

cv2.imshow('output_200_30', out5)
cv2.moveWindow('output_200_30', (np.size(img, 1))+10, (np.size(img, 0))+60)

cv2.imshow('output_11_60', out6)
cv2.moveWindow('output_11_60', (2*np.size(img, 1))+20, (np.size(img, 0))+60)

cv2.imshow('output_30_200', out7)
cv2.moveWindow('output_30_200', (3*np.size(img, 1))+30, (np.size(img, 0))+60)

# let the display screens stay
cv2.waitKey(0)