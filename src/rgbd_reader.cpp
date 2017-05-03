#include <kabamaru/rgbd_reader.hpp>
#include <opencv2/videoio/videoio_c.h>

RGBDReader::RGBDReader() {
	reader_type = RGBDReader::OPENNI_DEVICE;
}

RGBDReader::RGBDReader(const std::string &rgb_fmt, const std::string &depth_fmt) {
	rgb_file_fmt = rgb_fmt;
	depth_file_fmt = depth_fmt;
	reader_type = RGBDReader::FILE_READER;
}

RGBDReader::~RGBDReader() {
	release();
}

bool RGBDReader::open() {
	if (reader_type == RGBDReader::OPENNI_DEVICE) {
		return cap_openni.open(CV_CAP_OPENNI_ASUS);
	} else if (reader_type == RGBDReader::FILE_READER) {
		return cap_depth.open(depth_file_fmt, cv::CAP_IMAGES) && cap_rgb.open(rgb_file_fmt, cv::CAP_IMAGES);
	}
	return false;
}

void RGBDReader::release() {
	cap_depth.release();
	cap_rgb.release();
	cap_openni.release();
}

bool RGBDReader::getFrames(cv::Mat &rgb, cv::Mat &depth) {
	bool success = true;
	if (reader_type == RGBDReader::OPENNI_DEVICE) {
		success = success && cap_openni.grab();
		success = success && cap_openni.retrieve(depth, CV_CAP_OPENNI_DEPTH_MAP);
		success = success && cap_openni.retrieve(rgb, CV_CAP_OPENNI_BGR_IMAGE);
	} else if (reader_type == RGBDReader::FILE_READER) {
		success = success && cap_depth.grab();
		success = success && cap_rgb.grab();
		success = success && cap_depth.retrieve(depth);
		success = success && cap_rgb.retrieve(rgb);
	} else {
		success = false;
	}
	return success;
}
