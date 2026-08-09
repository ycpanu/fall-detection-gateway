#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
	std::cout << "Sysroot Test\n";
	std::cout << "OpenCV Version:" << CV_VERSION << std::endl;

	cv::Mat test_img = cv::Mat::zeros(100, 100, CV_8UC3);
	std::cout << "Successfully created a " << test_img.cols << "x" << test_img.rows
		  << " image matrix on Orange Pi NPU gateway\n";
	return 0;
}

