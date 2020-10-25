OPENCV_LIBS:=$(shell pkg-config --libs opencv)
CUDA_CXXFLAGS:=-I /usr/local/cuda/include
CUDA_LIBS:=-L /usr/local/cuda/lib64 -lcudart

SRC:=$(wildcard *.cpp)
EXE:=$(patsubst %.cpp, %, ${SRC})
CU_SRC:=$(wildcard *.cu)
CU_EXE:=$(patsubst %.cu, %, ${CU_SRC})

.PHONY: all

all: ${EXE} ${CU_EXE}

% : %.cpp
	g++ -o $@ $< ${OPENCV_LIBS}

% : %.cu
	nvcc -o $@ $< ${CUDA_CXXFLAGS} ${OPENCV_LIBS} ${CUDA_LIBS}