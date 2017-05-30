import cv2
import numpy as np

import time
import datetime
import matplotlib as plt
 
if __name__ == '__main__' :
 
    # Read source image.

    im_dst = cv2.imread("/home/sahil/Code/ComputerVision/Homography/IMG_20120708_180236.jpg",0)
    # im_src = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    # Four corners of the book in source image
 
 
    # Read destination image.
    im_src = cv2.imread("/home/sahil/Code/ComputerVision/Homography/IMG_20120718_182225.jpg",0)

    # cv2.rectangle(im_dst, (100, 100), (1948, 1436), (255,0,0), 2)

    # im_dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Four corners of the book in destination image.
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    s = time.time()
    kp1, des1 = sift.detectAndCompute(im_src,None)
    kp2, des2 = sift.detectAndCompute(im_dst,None)
    d = time.time()
    print ("TIME",(d-s))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    s = time.time()
    matches = flann.knnMatch(des1,des2,k=2)
    d = time.time()
    print ("TIME match:",(d-s))

    # store all the good matches as per Lowe's ratio test.
    good = []
    s = time.time()
    for m,n in matches:
        print m
        print n
        if m.distance < 0.7*n.distance:
            good.append(m)
    d = time.time()
    print ("TIME matching:",(d-s))

     
    # Calculate Homography
    MIN_MATCH_COUNT = 25
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        s = time.time()
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        d = time.time()
        print ("TIME homo:",(d-s))
        matchesMask = mask.ravel().tolist()

        h,w = im_src.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # pts2 = np.float32([ [100,100],[100,1436],[1948,1436],[1948,100] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        # dst2 = cv2.perspectiveTransform(pts2,M)

        # print np.int32(dst2)
    
        im_dst = cv2.polylines(im_dst,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.resize(im_dst,(512,512))


    cv2.imshow('gray',img3)
 
    cv2.waitKey(0)