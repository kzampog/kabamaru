#ifndef RGBD_READER_HPP
#define RGBD_READER_HPP

#include <string>
#include <opencv2/videoio.hpp>

class RGBDReader {
public:
	RGBDReader();
	RGBDReader(const std::string &rgb_fmt, const std::string &depth_fmt);
	~RGBDReader();

	bool open();
	void release();
	bool getFrames(cv::Mat &rgb, cv::Mat &depth);
private:
	enum ReaderType {OPENNI_DEVICE, FILE_READER};
	ReaderType reader_type;

	std::string depth_file_fmt;
	std::string rgb_file_fmt;

	cv::VideoCapture cap_depth;
	cv::VideoCapture cap_rgb;
	cv::VideoCapture cap_openni;
};

#endif /* RGBD_READER_HPP */
