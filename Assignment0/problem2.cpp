#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<stdio.h>

using namespace cv;
using namespace std;

int main() {
  VideoCapture cap(0);
  Mat edges;
  int framecount=0;
  for(;framecount<=100;) {
      Mat frame;
      cap>>frame;
      imshow("Webcam",frame);
      char filename[128];
      sprintf(filename,"webframe%06d.jpg",framecount);
      imwrite(filename,frame);
      framecount++;
      cout<<framecount<<endl;
      if(waitKey(30) == 27)
        break;
  }
  return 0;
}
