/* Includes */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include "rhorefc.h"
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv/cv.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv/highgui.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>



/* Namespaces */
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

/* Main */
int main(int argc, char* argv[]) {

    // Read source image.
    Mat img_object = imread("IMG_20120708_180236.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_scene = imread( "IMG_20120718_182225.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    // Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(100);
    Ptr<Feature2D> surf = SIFT::create(400);
    // SurfFeatureDetector detector( minHessian );

    vector<KeyPoint> keypoints_object, keypoints_scene;

    Mat descriptors_object, descriptors_scene;

    surf -> detectAndCompute( img_object,noArray(), keypoints_object, descriptors_object );
    surf -> detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

    // //-- Step 2: Calculate descriptors (feature vectors)
    // Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(ORB);



    // extractor -> compute( img_object, keypoints_object, descriptors_object );
    // extractor -> compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 200;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
    }

    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    vector<Point2f> src;
    vector<Point2f> dst;

    for( int i = 0; i < good_matches.size(); i++ ) {
      //-- Get the keypoints from the good matches
      src.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      dst.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat h = findHomography(src, dst, CV_RANSAC);

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); 
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); 
    obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, h);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    Size size(1024,512);//the dst image size,e.g.100x100
    Mat dstr;//dst image
    resize(img_matches,dstr,size);
    imshow("Warped Source Image", dstr);
    waitKey(0);

    // vector<Point2f> src; 	/* should be filled */
    // vector<Point2f> dst;	/* should be filled */
    int npoints = 100;

    Mat    tempMask, _H;
    bool   result;
    const double ransacReprojThreshold = 3;
    double beta = 0.35; /* 0.35 is a value that often works. */
    const int maxIters = 2000;
    const double confidence = 0.995;

    /* Create temporary output matrix (RHO outputs a single-precision H only). */
    Mat tmpH = Mat(3, 3, CV_32FC1);

    /* Create output mask. */
    tempMask = Mat(npoints, 1, CV_8U);

    /**
     * Make use of the RHO estimator API.
     *
     * This is where the math happens. A homography estimation context is
     * initialized, used, then finalized.
     */

    RHO_HEST_REFC* p = rhoRefCInit();

    /**
     * Optional. Ideally, the context would survive across calls to
     * findHomography(), but no clean way appears to exit to do so. The price
     * to pay is marginally more computational work than strictly needed.
     */

    rhoRefCEnsureCapacity(p, npoints, beta);

    /**
     * The critical call. All parameters are heavily documented in rhorefc.h.
     *
     * Currently, NR (Non-Randomness criterion) and Final Refinement (with
     * internal, optimized Levenberg-Marquardt method) are enabled.
     */

    result = !!rhoRefC(p,
                      (const float*)&src[0],
                      (const float*)&dst[0],
                      (char*)       tempMask.data,
                      (unsigned)    npoints,
                      (float)       ransacReprojThreshold,
                      (unsigned)    maxIters,
                      (unsigned)    maxIters,
                      confidence,
                      4U,
                      beta,
                      RHO_FLAG_ENABLE_NR | RHO_FLAG_ENABLE_FINAL_REFINEMENT,
                      NULL,
                      (float*)tmpH.data);

    /**
     * Cleanup.
     */

    rhoRefCFini(p);

    /* Convert float homography to double precision. */
    tmpH.convertTo(_H, CV_64FC1);

    /* Maps non-zero mask elems to 1, for the sake of the testcase. */
    for(int k=0;k<npoints;k++){
        tempMask.data[k] = !!tempMask.data[k];
    }
    
    //tempMask.copyTo(_tempMask);

    return result;
}
