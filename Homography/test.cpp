#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    {
        // Load an image
        Mat3b img = imread("IMG_20120718_182225.jpg");

        // Create the mask
        Mat1b mask(img.rows, img.cols, uchar(GC_PR_BGD));
        circle(mask, Point(img.cols / 2, img.rows / 2), 20, Scalar(GC_FGD), -1);

        Mat bgdModel, fgdModel;
        grabCut(img, mask, Rect(), bgdModel, fgdModel, 1);

        imshow("Mask", mask);
        waitKey(1);

        // Save model to file
        {
            FileStorage fs("mymodels.yml", FileStorage::WRITE);
            fs << "BgdModel" << bgdModel;
            fs << "FgdModel" << fgdModel;
        }
    }

    {
        // Load another image
        Mat3b img = imread("IMG_20120708_180236.jpg");

        // Load models from file
        Mat bgdModel, fgdModel;

        {
            FileStorage fs("mymodels.yml", FileStorage::READ);
            fs["BgdModel"] >> bgdModel;
            fs["FgdModel"] >> fgdModel;
        }

        // Create a mask
        Mat1b mask(img.rows, img.cols, uchar(GC_PR_BGD));
        circle(mask, Point(img.cols / 2, img.rows / 2), 20, Scalar(GC_FGD), -1);

        grabCut(img, mask, Rect(), bgdModel, fgdModel, 1);

        imshow("Other Mask", mask);
        waitKey(1);

    }
    return 0;
}