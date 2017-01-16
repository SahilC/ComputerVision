CC = g++
CFLAGS = -g -Wall
SRCS = chroma_keying.cpp
PROG = chroma_keying

PROB1 = problem1.cpp
PROG1 = problem1

PROB2 = problem2.cpp
PROG2 = problem2

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
		$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

$(PROG1):$(PROB1)
	          $(CC) $(CFLAGS) -o $(PROG1) $(PROB1) $(LIBS) 

$(PROG2):$(PROB2)
	          $(CC) $(CFLAGS) -o $(PROG2) $(PROB2) $(LIBS) 
