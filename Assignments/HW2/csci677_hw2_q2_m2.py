"""
Watershed Segmentor
Homework-2 Part-b (Method-2)
Concept reference: Reference PDF sent by Prof
Chinmay Chinara
USC ID: 2527237452
Date: 26th September 2017
"""

import numpy as np
import cv2

# Reading image
# img = cv2.imread('101085.jpg', 3)
# img = cv2.imread('300091.jpg', 3)
img = cv2.imread('253027.jpg', 3)

# display the input image
cv2.imshow('Input_Image', img)
cv2.moveWindow('Input_Image', 0, 0)

# save the height and width of the image
height, width, _ = img.shape
print(height)
print(width)

# Initialize the markers with zeros
markers = np.zeros((height,width), np.int32)

# logic for defining windows for defining region for markers
k = 1
for i in range(0,height):
    for j in range(0,width):
        if(i % 30 == 0 and j % 30 == 0):
            markers[i][j] = k
            k = k + 1

# watershed algorithm applied over the image
markers = cv2.watershed(img, markers)

# show the segmented lines over the image
img[markers == -1] = [255, 0, 0]
cv2.imshow('Output_segmented', img)
cv2.moveWindow('Output_segmented', np.size(img, 1)+10, 0)

for i in range(1,k):
    img[markers == i] = [np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255)]

# fill different colors over the segmented regions for better clarity
cv2.imshow('Output_color_segmented', img)
cv2.moveWindow('Output_color_segmented', (2*np.size(img, 1))+20, 0)

# wait for ESC key to exit
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()