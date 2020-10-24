OPENCV_LIBS=$(shell pkg-config --libs opencv)
SRC:=$(wildcard *.cpp)
EXE:=$(patsubst %.cpp, %, ${SRC})

.PHONY: all

all: ${EXE}

% : %.cpp
	g++ -o $@ $< ${OPENCV_LIBS}

