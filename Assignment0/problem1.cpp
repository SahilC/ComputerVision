#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat frame;
    VideoCapture cap("webframe%06d.jpg");
    int fps;
    if(argc > 1) {
        cout << argv[1]<<endl;
        fps = atoi(argv[1]);
    } else {
      fps = 30;
    }
    VideoWriter writer("webcam_created.avi", CV_FOURCC('M','J','P','G'), fps, Size(640, 480));
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
