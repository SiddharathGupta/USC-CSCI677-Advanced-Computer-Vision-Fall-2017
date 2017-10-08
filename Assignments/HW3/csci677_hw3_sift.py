"""
SIFT Features
Homework-3
based on concept reference: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
Chinmay Chinara
USC ID: 2527237452
Date: 6th October 2017
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img_detect = 'image_2' # line 5
img_target = 'image_5' # line 6

# Read the images
img1 = cv2.imread(img_detect + '.jpg',0) # queryImage
img2 = cv2.imread(img_target + '.jpg',0) # trainImage

# Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print('The features found in querry image ' + img_detect + ': ' + str(des1.shape[0]))
print('The features found in train image ' + img_target + ': ' + str(des2.shape[0]))

# features overlaid on the images with circles based on size of keypoints and its orientation
img4 = cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5 = cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT keypoints and orientation querry image',img4)
cv2.imshow('SIFT keypoints and orientation train image',img5)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Sort the good matches in the order of their distance and plot the top 20 matches
good = sorted(good, key = lambda x:x.distance)
img6 = cv2.drawMatches(img1, kp1, img2, kp2, good[:20], None, flags=2)
cv2.imshow('Top 20 matches between ' + img_detect + ' and ' + img_target, img6)

# set that atleast 10 matches are to be there to find the objects
MIN_MATCH_COUNT = 10

# condition set that atleast 10 matches are to be there to find the objects
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # compute the homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    print('The total number of good matches are: ' + str(mask.size))
    print('The total number of inliers (good matches providing correct estimation and total numbers consistent with the computed homography)) are: ' + str(np.sum(matchesMask)))
    print('The homography matrix when ' + img_detect + ' is matched with ' + img_target + ':')
    print(M)

    # capture the height and width of the image
    h, w = img1.shape

    # If enough matches are found, we extract the locations of matched keypoints in both the images
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # The locations of matched keypoints  are passed to find the perpective transformation
    dst = cv2.perspectiveTransform(pts,M)

    # Once we get this 3x3 transformation matrix, we use it to transform the corners of queryImage to corresponding points in trainImage.
    # Then we draw it.
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ('Not enough matches are found - %d/%d' % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

# Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed)
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

# draw the final SIFT matching
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
cv2.imshow('Final SIFT Match between ' + img_detect + ' and ' + img_target, img3)

# wait for ESC key to exit
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
    plt.close('all')