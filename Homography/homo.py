import os
import cv2
import numpy as np

import time
import datetime
import matplotlib as plt
 
if __name__ == '__main__' :
 
    # Read source image.

    file_dir = '/home/sahil/Data/Golconda Photos/taramati_mosque'
    im_src = cv2.imread('/home/sahil/Data/Golconda Photos/taramati_mosque'+"/IMG_20120718_182225.jpg",0)
    sift = cv2.xfeatures2d.SURF_create()

    kp1, des1 = sift.detectAndCompute(im_src,None)
    num = 1
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    for dirpath, dnames, fnames in os.walk(file_dir):
        for f in fnames:
            im_dst = cv2.imread("/home/sahil/Data/Golconda Photos/taramati_mosque/"+f,0)
            im_dst2 = cv2.imread("/home/sahil/Data/Golconda Photos/taramati_mosque/"+f)
            # cv2.imshow("GREY",im_dst)
            # im_src = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
            # Four corners of the book in source image
    
        
            # Read destination image.
            

            # cv2.rectangle(im_dst, (100, 100), (1948, 1436), (255,0,0), 2)

            # im_dst = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Four corners of the book in destination image.
            # Initiate SIFT detector
            

            # find the keypoints and descriptors with SIFT
    
            kp2, des2 = sift.detectAndCompute(im_dst,None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

     
            # Calculate Homography
            MIN_MATCH_COUNT = 25
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
                matchesMask = mask.ravel().tolist()

                h,w = im_src.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                # pts2 = np.float32([ [100,100],[100,1436],[1948,1436],[1948,100] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                # dst2 = cv2.perspectiveTransform(pts2,M)

                # print np.int32(dst2)
                
                mask = np.zeros(im_dst.shape,np.uint8)
                cv2.fillPoly(mask,[np.int32(dst)],1)

                _, contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                # x,y,w,h = cv2.boundingRect(contours[0])

                mask, bgdModel, fgdModel = cv2.grabCut(im_dst2, mask, None, bgdModel, fgdModel, 35, cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                im_dst = im_dst*mask

                # im_dst = cv2.polylines(im_dst,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
            else:
                print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
                matchesMask = None

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

            img3 = cv2.resize(im_dst,(512,512))

            num += 1

            cv2.imwrite('gray'+str(num)+'.png',img3)
            #cv2.waitKey(0)