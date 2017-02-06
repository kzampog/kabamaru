#include <sisyphus/io_utilities.hpp>

std::string cvTypeToString(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}
	r += "C";
	r += (chans+'0');

	return r;
}

void readSIFTModelFromFile(const std::string &fname, Eigen::MatrixXf &loc, std::vector<float> &descr) {
	readEigenMatrixFromFile(fname, loc);
	readVectorFromFile(fname, descr);

	// int N;
	// std::ifstream infile(fname, std::ifstream::binary);
	// infile.read((char*)(&N), sizeof(int));
	
	// std::vector<float> loc_vec(3*N);
	// descr = std::vector<float>(128*N);

	// infile.read((char*)(&loc_vec[0]), 3*N*sizeof(float));
	// infile.read((char*)(&descr[0]), 128*N*sizeof(float));
	// infile.close();

	// loc = Eigen::Map<Eigen::MatrixXf>(&loc_vec[0],3,N);
}

void writeSIFTModelToFile(const std::string &fname, const Eigen::MatrixXf &loc, const std::vector<float> &descr) {
	writeEigenMatrixToFile(fname, loc);
	writeVectorToFile(fname, descr);

	// int N = loc.cols();
	// std::vector<float> loc_vec(loc.data(), loc.data() + loc.rows() * loc.cols());

	// std::ofstream outfile(fname, std::ofstream::binary);
	// outfile.write((char*)(&N), sizeof(int));
	// outfile.write((char*)(&loc_vec[0]), 3*N*sizeof(float));
	// outfile.write((char*)(&descr[0]), 128*N*sizeof(float));
	// outfile.close();
}

void readStereoRigParametersFromXMLFile(const std::string &fname, cv::Mat &K0, cv::Mat &d0, cv::Size &im0_size, cv::Mat &K1, cv::Mat &d1, cv::Size &im1_size, cv::Mat &R, cv::Mat &t) {

	cv::FileStorage fs("/home/kzampog/Desktop/calibration.xml", cv::FileStorage::READ);

	cv::FileNode stereo_calib_node = fs["Rig"];
	cv::FileNode cameras_node = stereo_calib_node["Cameras"];
	cv::FileNode camera_0_node = cameras_node[0];
	cv::FileNode camera_1_node = cameras_node[1];
	cv::FileNode intrinsics_0_node = camera_0_node["Intrinsics"];
	cv::FileNode intrinsics_1_node = camera_1_node["Intrinsics"];
	cv::FileNode extrinsics_node = camera_1_node["Extrinsics"];

	intrinsics_0_node["intrinsic_mat"] >> K0;
	intrinsics_0_node["distortion_coeffs"] >> d0;
	intrinsics_1_node["intrinsic_mat"] >> K1;
	intrinsics_1_node["distortion_coeffs"] >> d1;
	extrinsics_node["rotation"] >> R;
	extrinsics_node["translation"] >> t;

	im0_size = cv::Size(static_cast<int>(intrinsics_0_node["resolution"]["width"]), static_cast<int>(intrinsics_0_node["resolution"]["height"]));

	im1_size = cv::Size(static_cast<int>(intrinsics_1_node["resolution"]["width"]), static_cast<int>(intrinsics_1_node["resolution"]["height"]));

	// K0.convertTo(K0, CV_32F);
	// d0.convertTo(d0, CV_32F);
	// K1.convertTo(K1, CV_32F);
	// d1.convertTo(d1, CV_32F);
	// R.convertTo(R, CV_32F);
	// t.convertTo(t, CV_32F);
}
