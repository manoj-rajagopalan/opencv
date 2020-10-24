#include <iostream>
#include <cassert>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
	assert(argc == 2);
	std::cout << "Reading from " << argv[1] << std::endl;
	cv::Mat img = cv::imread(argv[1]);
	std::cout << "img.shape = " << img.cols << 'x' << img.rows << std::endl;

	cv::Mat img_box_filter;
	cv::boxFilter(img, img_box_filter, -1, cv::Size(9,9), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);

	std::string output_filename(argv[1]);
	auto dot_pos = output_filename.rfind(".");
	output_filename = output_filename.substr(0, dot_pos) + "-box_filter" + output_filename.substr(dot_pos);
	std::cout << "Writing to " << output_filename << std::endl;
	const bool ok = cv::imwrite(output_filename, img_box_filter);
	assert(ok);
	return 0;
}

