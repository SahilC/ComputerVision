CC = g++
CFLAGS = -g -Wall
SRCS = example.cpp rhorefc.cpp
PROG = example

# OPENCV = `pkg-config opencv --cflags --libs`
# LIBS = $(OPENCV)
# LIBPATH =-L../../opencv/debug/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc  -L/usr/local/lib 

$(PROG):$(SRCS)
		$(CC) $^ -o $@ $(LIBS)
