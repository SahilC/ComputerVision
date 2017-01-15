#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

int main() {
    Mat frame;
    VideoCapture cap("frame_%06d.jpg");

    VideoWriter writer("created.avi", CV_FOURCC('M','J','P','G'), 30, Size(640, 360));
    int i = 0;
    while(1) {
        cout << i << endl;
        i++;
        cap >> frame;
        if (frame.empty())
            break;
        writer.write(frame);
    }
    return 0;
}
