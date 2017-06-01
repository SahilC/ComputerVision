#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( )
{
 // Open another image
    Mat image, img_1;
    image = imread("IMG_20120718_182225.jpg");

    // define bounding rectangle 
    Rect rectangle(50,70,image.cols-150,image.rows-180);

    Mat result; // segmentation result (4 possible values)
    Mat bgModel,fgModel; // the models (internally used)

    // GrabCut segmentation
    grabCut(image,    // input image
                    result,   // segmentation result
                            rectangle,// rectangle containing foreground 
                            bgModel,fgModel, // models
                            1,        // number of iterations
                            GC_INIT_WITH_RECT); // use rectangle

    // Get the pixels marked as likely foreground
    compare(result,GC_PR_FGD,result,CMP_EQ);
    // Generate output image
    Mat foreground(image.size(),CV_8UC3,Scalar(255,255,255));
    image.copyTo(foreground,result); // bg pixels not copied


    // draw rectangle on original image
   //  rectangle(image, rectangle, Scalar(255,255,255),1);
    Size size(1024,512);//the dst image size,e.g.100x100
  	// Mat dstr2;//dst image
  	// resize(image,dstr2,size);
  	// imshow("Warped Source1 Image", dstr2);
    // namedWindow("Image");
    // imshow("Image",image);

    // display result
  	Mat dstr;//dst image
  	resize(foreground,dstr,size);
  	imshow("Warped Source Image", dstr);
  	waitKey(0);
    // namedWindow("Segmented Image");
    // imshow("Segmented Image",foreground);





    img_1 = imread("IMG_20120708_180236.jpg");

    // define bounding rectangle 
    Mat mask1;
    mask1.create( img_1.size(), CV_8UC1);
    mask1.setTo( GC_PR_FGD );
    Rect rect1;
    rect1.x=0;
    rect1.y=0;
    rect1.width=0;//img.cols;
    rect1.height=0;//img.rows;
    (mask1(rect1)).setTo( Scalar(GC_BGD) );

    circle(mask1, Point(img_1.cols / 2, img_1.rows / 2), 20, Scalar(GC_FGD), -1);


    grabCut(img_1, mask1, rect1, bgModel, fgModel, 1, GC_EVAL);

    // Get the pixels marked as likely foreground
    compare(mask1,GC_PR_FGD,mask1,CMP_EQ);
    // Generate output image
    Mat foreground2(img_1.size(),CV_8UC3,Scalar(255,255,255));
    img_1.copyTo(foreground2,mask1); // bg pixels not copied


    // draw rectangle on original image
   //  rectangle(image, rectangle, Scalar(255,255,255),1);
  	// Mat dstr2;//dst image
  	// resize(image,dstr2,size);
  	// imshow("Warped Source1 Image", dstr2);
    // namedWindow("Image");
    // imshow("Image",image);

    // display result
  	resize(foreground2,dstr,size);
  	imshow("Warped Source Image", dstr);


    waitKey(0);
    return 0;
}