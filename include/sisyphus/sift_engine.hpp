#ifndef SIFT_ENGINE_HPP
#define SIFT_ENGINE_HPP

#include <SiftGPU.h>
#include <opencv2/core/core.hpp>

class SIFTEngine {
private:
	SiftGPU sift;
	SiftMatchGPU matcher;
	int match_buff[4096][2];
public:
	SIFTEngine();
	void detectFeatures(const cv::Mat &img, std::vector<SiftGPU::SiftKeypoint> &keys, std::vector<float> &descr);
	void getCorrespondences(const std::vector<float> &src_descr, const std::vector<float> &dst_descr, std::vector<int> &src_ind, std::vector<int> &dst_ind);
};

#endif /* SIFT_ENGINE_HPP */
