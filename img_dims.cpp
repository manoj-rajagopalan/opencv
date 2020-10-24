#include <iostream>
#include <cassert>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char *argv[])
{
	assert(argc == 2);
	cv::Mat img = cv::imread(argv[1]);
	std::cout << "img.shape = " << img.cols << 'x' << img.rows << std::endl;
	std::cout << "img.dims = " << img.dims << std::endl;
	std::cout << "img.elemSize = " << img.elemSize() << std::endl;
	std::cout << "img.depth (code) = " << img.depth() << std::endl;
	return 0;
}

