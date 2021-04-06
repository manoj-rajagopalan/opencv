UNAME=$(shell uname)

CXXFLAGS= -std=c++14 -g

ifeq (${UNAME},Darwin)
CXX:=g++
OPENCV_CXXFLAGS:= $(shell pkg-config --cflags opencv4)
OPENCV_LIBS:= $(shell pkg-config --libs opencv4)
CUDA_CXXFLAGS:=
CUDA_LIBS:=
else
CXX:=nvcc
OPENCV_CXXFLAGS:= $(shell pkg-config --cflags opencv)
OPENCV_LIBS:= $(shell pkg-config --libs opencv)
CUDA_CXXFLAGS:=-I /usr/local/cuda/include
CUDA_LIBS:=-L /usr/local/cuda/lib64 -lcudart
endif


SRC:=$(wildcard *.cpp)
EXE:=$(patsubst %.cpp, %, ${SRC})
CU_SRC:=$(wildcard *.cu)
CU_EXE:=$(patsubst %.cu, %, ${CU_SRC})

.PHONY: all clean

all: ${EXE} ${CU_EXE}

clean:
	rm -f *.o *.a ${EXE} ${CU_EXE}

%.o : %.cpp
	g++ -o $@ -c $< ${CXXFLAGS} ${OPENCV_CXXFLAGS}

%.o : %.cu
	nvcc -o $@ -dc $< ${CXXFLAGS} ${OPENCV_CXXFLAGS} ${CUDA_CXXFLAGS}

libmoving_object.a: moving_object.o shapes.o
	ar crs $@ $^
	ranlib $@

video_frame_generator: video_frame_generator.o libmoving_object.a
	${CXX} -o $@ $^ ${OPENCV_LIBS} ${CUDA_LIBS}

video_frame_generator_cuda: video_frame_generator_cuda.o libmoving_object.a
	${CXX} -o $@ $^ ${OPENCV_LIBS} ${CUDA_LIBS}

% : %.o
	${CXX} -o $@ $< ${OPENCV_LIBS} ${CUDA_LIBS}