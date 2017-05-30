CC = g++
CFLAGS = -g -Wall
SRCS = untitled.cpp rhorefc.cpp
PROG = example

OPENCV = `pkg-config --cflags --libs opencv`
# LIBS = $(OPENCV)
# LIBPATH =-L../../opencv/debug/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc  -L/usr/local/lib 

$(PROG):$(SRCS)
		$(CC) $^ -o $@ $(LIBS) $(OPENCV)
