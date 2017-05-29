#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
// #include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

using namespace std;
 
int main( int argc, char** argv)
{
    // Read source image.
    Mat img_object = imread("IMG_20120718_164616.jpg");
    Mat img_scene = imread( "IMG_20120718_164619.jpg");

  	if( !img_object.data || !img_scene.data )
  	{ 

  	// int minHessian = 400;

  	Ptr<ORB> detector = ORB::create(100, 2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
  	// SurfFeatureDetector detector( minHessian );

  	vector<KeyPoint> keypoints_object, keypoints_scene;

  	Mat descriptors_object, descriptors_scene;

  	detector -> detectAndCompute( img_object,noArray(), keypoints_object, descriptors_object );
  	detector -> detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );

  	// //-- Step 2: Calculate descriptors (feature vectors)
  	// Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(ORB);
	
  	

  	// extractor -> compute( img_object, keypoints_object, descriptors_object );
  	// extractor -> compute( img_scene, keypoints_scene, descriptors_scene );

  	//-- Step 3: Matching descriptor vectors using FLANN matcher
  	FlannBasedMatcher matcher;
  	vector< DMatch > matches;
  	matcher.match( descriptors_object, descriptors_scene, matches );

  	double max_dist = 0; double min_dist = 100;

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

  	vector<Point2f> obj;
  	vector<Point2f> scene;

  	for( int i = 0; i < good_matches.size(); i++ )
  	{
  	  //-- Get the keypoints from the good matches
  	  obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
  	  scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  	}

    // Four corners of the book in source image
    // vector<Point2f> pts_src;
    // pts_src.push_back(Point2f(141, 131));
    // pts_src.push_back(Point2f(480, 159));
    // pts_src.push_back(Point2f(493, 630));
    // pts_src.push_back(Point2f(64, 601));
 
 
    // Read destination image.
    // Mat im_dst = imread();
    // Four corners of the book in destination image.
    // vector<Point2f> pts_dst;
    // pts_dst.push_back(Point2f(318, 256));
    // pts_dst.push_back(Point2f(534, 372));
    // pts_dst.push_back(Point2f(316, 670));
    // pts_dst.push_back(Point2f(73, 473));
 
    // Calculate Homography
    Mat H = findHomography(obj, scene);
 
    // Output image
    // Mat im_out;
    // // Warp source image to destination based on homography
    // warpPerspective(img_scene, im_out, h, img_object.size());
 
    // // Display images
    // imshow("Source Image", img_scene);
    // imshow("Destination Image", img_object);
    // imshow("Warped Source Image", im_out);
     std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  // line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  // line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );
 
    waitKey(0);
}
}