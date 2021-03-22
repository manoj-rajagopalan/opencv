#include <iostream>
#include <cassert>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/calib3d.hpp>

int main(int argc, char *argv[])
{
	assert(argc == 3 && "Usage: stereo <left_image> <right_image>");
	
	std::cout << "Reading left image from " << argv[1] << std::endl;
	cv::Mat l_img = cv::imread(argv[1]);
	std::cout << "l_img.shape = " << l_img.cols << 'x' << l_img.rows << std::endl;

	std::cout << "Reading right image from " << argv[2] << std::endl;
	cv::Mat r_img = cv::imread(argv[2]);
	std::cout << "r_img.shape = " << r_img.cols << 'x' << r_img.rows << std::endl;

	cv::UMat disparity(l_img.rows, l_img.cols, CV_8UC1);
	auto stereo = cv::StereoSGBM::create(0, 64, 5);
	stereo->compute(l_img, r_img, disparity);

	std::string output_filename("disparity.png");
	std::cout << "Writing disparity to " << output_filename << std::endl;
	const bool ok = cv::imwrite(output_filename, disparity);
	assert(ok);
	return 0;
}

