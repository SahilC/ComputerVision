#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( )
{
 // Open another image
    Mat image;
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
    // namedWindow("Segmented Image");
    // imshow("Segmented Image",foreground);


    waitKey(0);
    return 0;
}