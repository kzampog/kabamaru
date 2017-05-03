#ifndef IO_UTILITIES_HPP
#define IO_UTILITIES_HPP

#include <fstream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

std::string cvTypeToString(int type);

template<class Matrix>
void readEigenMatrixFromFile(const std::string &filename, Matrix& matrix){
	std::ifstream in(filename, std::ifstream::binary);
	typename Matrix::Index rows = 0, cols = 0;
	in.read((char*)(&rows), sizeof(typename Matrix::Index));
	in.read((char*)(&cols), sizeof(typename Matrix::Index));
	matrix.resize(rows, cols);
	in.read((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
	in.close();
}

template<class Matrix>
void writeEigenMatrixToFile(const std::string &filename, const Matrix& matrix){
	std::ofstream out(filename, std::ofstream::binary);
	typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
	out.write((char*)(&rows), sizeof(typename Matrix::Index));
	out.write((char*)(&cols), sizeof(typename Matrix::Index));
	out.write((char*)matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
	out.close();
}

template<typename ScalarT>
void readVectorFromFile(const std::string &filename, std::vector<ScalarT> &vec) {
	Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec_e;
	readEigenMatrixFromFile(filename, vec_e);
	vec.resize(vec_e.size());
	Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>::Map(&vec[0], vec_e.size()) = vec_e;
}

template<typename ScalarT>
void writeVectorToFile(const std::string &filename, const std::vector<ScalarT> &vec) {
	Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec_e = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>::Map(&vec[0], vec.size());
	writeEigenMatrixToFile(filename, vec_e);
}

void readSIFTModelFromFile(const std::string &fname, Eigen::MatrixXf &loc, std::vector<float> &descr);

void writeSIFTModelToFile(const std::string &fname, const Eigen::MatrixXf &loc, const std::vector<float> &descr);

void readStereoRigParametersFromXMLFile(const std::string &fname, cv::Mat &K0, cv::Mat &d0, cv::Size &im0_size, cv::Mat &K1, cv::Mat &d1, cv::Size &im1_size, cv::Mat &R, cv::Mat &t);

#endif /* IO_UTILITIES_HPP */
