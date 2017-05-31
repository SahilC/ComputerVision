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
    
    int minHessian = 400;
    Ptr<Feature2D> surf = SURF::create();
    // SurfFeatureDetector detector( minHessian );

    vector<KeyPoint> keypoints_object, keypoints_scene;

    Mat descriptors_object, descriptors_scene;

    surf -> detectAndCompute( img_object,noArray(), keypoints_object, descriptors_object );
    surf -> detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );


    if( !img_object.data || !img_scene.data ) { 
      std::cout<< " --(!) Error reading images " << std::endl; return -1; 
    }


    //-- Step 3: Matching descriptor vectors using FLANN matcher
    BFMatcher matcher(NORM_L2,false);
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 300;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    cout<<matches[i].distance<<endl;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat mask;
  Mat H = findHomography( obj, scene, CV_RANSAC, 1.0, mask);

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  // imshow( "Good Matches & Object detection", img_matches );
   Size size(1024,512);//the dst image size,e.g.100x100
   Mat dstr;//dst image
   resize(img_matches,dstr,size);
   imshow("Warped Source Image", dstr);
   waitKey(0);

  waitKey(0);
  return 0;
}
