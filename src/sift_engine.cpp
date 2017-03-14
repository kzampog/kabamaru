#include <iostream>
#include <GL/gl.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <sisyphus/sift_engine.hpp>

SIFTEngine::SIFTEngine() {
	matcher = SiftMatchGPU(4096);
	char* sift_params[] = {(char*)"fo", (char*)"-1", (char*)"-v", (char*)"0", (char*)"-s", (char*)"-m", (char*)"1", (char*)"-pack", (char*)"-glsl"};
	sift.ParseParam(sizeof(sift_params)/sizeof(char*), sift_params);
	if (sift.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
		std::cout << "SiftGPU error" << std::endl;
		throw;
	}
	if (matcher.VerifyContextGL() == 0) {
		std::cout << "SiftMatchGPU error" << std::endl;
		throw;
	}
	// std::cout << "SIFTEngine object created!" << std::endl;
}

void SIFTEngine::detectFeatures(const cv::Mat &img, std::vector<SiftGPU::SiftKeypoint> &keys, std::vector<float> &descr) {
	cv::Mat gray;
	if (img.channels() > 1) {
		cv::cvtColor(img, gray, CV_BGR2GRAY);
	} else {
		gray = img;
	}
	sift.RunSIFT(gray.cols, gray.rows, gray.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
	int num_feat = sift.GetFeatureNum();
	keys = std::vector<SiftGPU::SiftKeypoint>(num_feat);
	descr = std::vector<float>(num_feat * 128);
	sift.GetFeatureVector(&keys[0], &descr[0]);
}

void SIFTEngine::getCorrespondences(const std::vector<float> &src_descr, const std::vector<float> &dst_descr, std::vector<int> &src_ind, std::vector<int> &dst_ind) {
	matcher.SetDescriptors(0, src_descr.size()/128, &src_descr[0]);
	matcher.SetDescriptors(1, dst_descr.size()/128, &dst_descr[0]);
	int num_match = matcher.GetSiftMatch(4096, match_buff);
	src_ind = std::vector<int>(num_match);
	dst_ind = std::vector<int>(num_match);
	for (int j = 0; j < num_match; j++) {
		src_ind[j] = match_buff[j][0];
		dst_ind[j] = match_buff[j][1];
	}
}
