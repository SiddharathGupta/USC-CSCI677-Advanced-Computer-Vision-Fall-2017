"""
Watershed Segmentor
Homework-2 Part-b (Method-3)
Concept reference: Using mouse clicks to capture marker positions
Chinmay Chinara
USC ID: 2527237452
Date: 26th September 2017
"""

import numpy as np
import cv2

# Reading image
# img = cv2.imread('101085.jpg', 3)
img = cv2.imread('300091.jpg', 3)
# img = cv2.imread('253027.jpg', 3)

# make a copy of the image for internal use
img1 = img.copy()

print('Double click on the region in the Input image you want to position a marker !!!')

# display the input image
cv2.imshow('Input_Image', img)
cv2.moveWindow('Input_Image', 0, 0)

# Add one to all labels so that sure background is not 0, but 1
height, width, _ = img.shape

# Now, mark the region of unknown with zero
markers = np.zeros((height,width), np.int32)

# mouse click event to capture marker positions
def mouse_click(event, x, y, flags, params):
    if(event == cv2.EVENT_LBUTTONDBLCLK):
        cv2.circle(img1, (x,y),3,(0,0,255),-1)
        markers[y][x] = np.random.randint(0,10000)
        print('y: ' + str(y) +', x: ' + str(x))
        print('Press ESC key once you are done !!!')

# logic to capture locations from image popped up
cv2.setMouseCallback('Input_Image', mouse_click)
while(1):
    cv2.imshow('Input_Image', img1)
    if cv2.waitKey(20) & 0xff == 27:
        break

# watershed algorithm applied over the image
markers = cv2.watershed(img, markers)

# show the segmented lines over the image
img[markers == -1] = [0, 0, 255]
cv2.imshow('Output_segmented', img)
cv2.moveWindow('Output_segmented', np.size(img, 1)+10, 0)

for i in range(0, 10000):
    img[markers == i] = [np.random.randint(0,255), np.random.randint(0,255),np.random.randint(0,255)]

# fill different colors over the segmented regions for better clarity
cv2.imshow('Output_color_segmented', img)
cv2.moveWindow('Output_color_segmented', (2*np.size(img, 1))+20, 0)

# wait for ESC key to exit
k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()