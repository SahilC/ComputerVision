
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

Scalar bgr2ycrcb( Scalar bgr ) {
	double R = bgr[ 2 ];
	double G = bgr[ 1 ];
	double B = bgr[ 0 ];
	double delta = 128; // Note: change this value if image type isn't CV_8U.

	double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
	double Cr = ( R - Y ) * 0.713 + delta;
	double Cb = ( B - Y ) * 0.564 + delta;

	return Scalar( Y, Cr, Cb, 0 /* ignored */ );
}

Mat1b chromaKey( const Mat3b & imageBGR, Scalar chromaBGR, double tInner, double tOuter ) {

	assert( tInner <= tOuter );

	assert( ! imageBGR.empty() );
	Size imageSize = imageBGR.size();
	Mat3b imageYCrCb;
	cvtColor( imageBGR, imageYCrCb, COLOR_BGR2YCrCb );
	Scalar chromaYCrCb = bgr2ycrcb( chromaBGR ); // Convert a single BGR value to YCrCb.

	Mat1b mask = Mat1b::zeros( imageSize );
	const Vec3d key( chromaYCrCb[ 0 ], chromaYCrCb[ 1 ], chromaYCrCb[ 2 ] );

	for ( int y = 0; y < imageSize.height; ++y ) {
		for ( int x = 0; x < imageSize.width; ++x ) {
			const Vec3d color( imageYCrCb( y, x )[ 0 ], imageYCrCb( y, x )[ 1 ], imageYCrCb( y, x )[ 2 ] );
			double distance = norm( key - color );

			if ( distance < tInner ) {
				// Current pixel is fully part of the background.
				mask( y, x ) = 0;
			} else if ( distance > tOuter ) {
				mask( y, x ) = 255;
			}	else {
				double d1 = distance - tInner;
				double d2 = tOuter   - tInner;
				uint8_t alpha = static_cast< uint8_t >( 255. * ( d1 / d2 ) );

				mask( y, x ) = alpha;
			}
		}
	}

	return mask;
}

Mat3b replaceBackground( const Mat3b & image, const Mat1b & mask, const Mat3b & image_bkgrd ) {
	Size imageSize = image.size();
	// const Vec3b bgColorVec( bgColor[ 0 ], bgColor[ 1 ], bgColor[ 2 ] );
	Mat3b newImage( image.size() );

	for ( int y = 0; y < imageSize.height; ++y ) {
		for ( int x = 0; x < imageSize.width; ++x ) {
			uint8_t maskValue = mask( y, x );

			if ( maskValue >= 255 ) {
				newImage( y, x ) = image( y, x );
			} else if ( maskValue <= 0 ) {
				newImage( y, x ) = image_bkgrd(y,x);
			} else {
				double alpha = 1. / static_cast< double >( maskValue );
				newImage( y, x ) = alpha * image( y, x ) + ( 1. - alpha ) * image_bkgrd(y,x);
			}
		}
	}

	return newImage;
}

int mixVideos() {
    string inputFilename_back = "forest.jpg";
    Mat3b input_bck = imread( inputFilename_back, IMREAD_COLOR );
    VideoCapture cap("/home/sahil/Code/ComputerVision/Assignment0/videos/dino.mp4");

    VideoCapture cap2("/home/sahil/Code/ComputerVision/Assignment0/videos/2d.mp4");
    if( !cap.isOpened() && !cap2.isOpened()){
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    double count = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

    while(1) {
        Mat frame , background;
        cap.read(frame);
        cap2.read(background);
        if(background.empty()) {
          cap2.set(CV_CAP_PROP_POS_MSEC, 0);
          cap2.read(background);
        }
        if(frame.empty())
           break;

        // resize(input_bck, input_bck, frame.size());
        resize(background, background, frame.size());
        Scalar chroma( 0, 255, 0, 0 );
       	double tInner = 100.;
       	double tOuter = 120.;
       	Mat1b mask = chromaKey( frame, chroma, tInner, tOuter );

       	Mat3b newBackground = replaceBackground( frame, mask, background);
        imshow("MyVideo", newBackground);
        if(waitKey(10) == 27) {
            break;
        }
    }
    return 0;
}

int main() {
  mixVideos();
	return 0;
}
