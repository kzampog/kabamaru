#ifndef SISYPHUS_REGISTRATION_UTILITIES_HPP
#define SISYPHUS_REGISTRATION_UTILITIES_HPP

#include <pcl/filters/extract_indices.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <sisyphus/sift_engine.hpp>

Eigen::Matrix4f homogeneousTransformationMatrixFromRt(const Eigen::Matrix3f &R, const Eigen::Vector3f &t);

int rigidlyAlignPoseSets(const std::vector<Eigen::Matrix4f> &src_poses, const std::vector<Eigen::Matrix4f> &dst_poses, Eigen::Matrix4f &tform);

Eigen::VectorXf transferErrorRigid3D(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst, const Eigen::Matrix3f &R, const Eigen::Vector3f &t);

int estimateRigidTransform3D(const Eigen::MatrixXf &src_in, const Eigen::MatrixXf &dst_in, Eigen::Matrix3f &R, Eigen::Vector3f &t);

int estimateRigidTransform3D(const Eigen::MatrixXf &src_in, const Eigen::MatrixXf &dst_in, Eigen::Matrix4f &tform);

int estimateRigidTransform3DRANSAC(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst, Eigen::Matrix3f &R, Eigen::Vector3f &t, std::vector<int> &inliers, int iter, int test_size, float dist_thresh, int inlier_thresh);

int estimateRigidTransform3DRANSAC(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst, Eigen::Matrix4f& tform, std::vector<int>& inliers, int iter, int test_size, float dist_thresh, int inlier_thresh);

Eigen::Vector3f projectiveToRealWorld(const Eigen::Vector3f &pt_pr, const Eigen::Matrix3f &K);

Eigen::Vector3f realWorldToProjective(const Eigen::Vector3f &pt_rw, const Eigen::Matrix3f &K);

void transformConvexPolytope(std::vector<Eigen::Vector4f> &polytope, const Eigen::Matrix4f &tform);

Eigen::MatrixXf extractMatrixColumnsFromIndices(const Eigen::MatrixXf &mat, const std::vector<int> &ind);

std::vector<float> extractSIFTDescriptorsFromIndices(const std::vector<float> &descr, const std::vector<int> &ind);

Eigen::MatrixXf extractSIFTKeypoint2DCoordinates(const std::vector<SiftGPU::SiftKeypoint> &keys);

int estimateRigidTransformSIFT3DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix3f &R, Eigen::Vector3f &t, std::vector<int> &inliers);

int estimateRigidTransformSIFT3DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix4f &tform, std::vector<int> &inliers);

int estimateRigidTransformSIFT2DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, const cv::Mat &K, const cv::Mat &d, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix3f &R, Eigen::Vector3f &t, std::vector<int> &inliers);

int estimateRigidTransformSIFT2DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, const cv::Mat &K, const cv::Mat &d, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix4f &tform, std::vector<int> &inliers);

template<typename PointT>
void extractKeypoint3DCoordinatesFromOrganizedPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, Eigen::MatrixXf &loc2d, std::vector<float> &descr, Eigen::MatrixXf &loc3d) {

	int num = loc2d.cols();

	std::vector<float> locv(3*num);
	std::vector<int> ind(num);
	int k = 0, r, c;
	PointT pt;

	for (int i = 0; i < num; i++) {
		c = std::round(loc2d(0,i));
		r = std::round(loc2d(1,i));
		pt = cloud->at(c,r);
		if (pt.z > 0) {
			locv[3*k] = pt.x;
			locv[3*k+1] = pt.y;
			locv[3*k+2] = pt.z;
			ind[k] = i;
			k++;
		}
	}
	locv.resize(3*k);
	ind.resize(k);

	loc2d = extractMatrixColumnsFromIndices(loc2d, ind);
	descr = extractSIFTDescriptorsFromIndices(descr, ind);
	loc3d = Eigen::Map<Eigen::MatrixXf>(&locv[0],3,k);

	return;
}

template<typename PointT>
cv::Mat organizedPointCloudToRGBImage(const typename pcl::PointCloud<PointT>::ConstPtr &cloud) {
	cv::Mat img;
	if (cloud->isOrganized()) {
		img = cv::Mat(cloud->height, cloud->width, CV_8UC3);
		if (!cloud->empty()) {
			for (int h = 0; h < img.rows; h++) {
				for (int w = 0; w < img.cols; w++) {
					Eigen::Vector3i rgb = cloud->at(w,h).getRGBVector3i();
					img.at<cv::Vec3b>(h,w)[0] = rgb[2];
					img.at<cv::Vec3b>(h,w)[1] = rgb[1];
					img.at<cv::Vec3b>(h,w)[2] = rgb[0];
				}
			}
		}
	}
	return img;
}

template<typename PointT>
cv::Mat organizedPointCloudToDepthImage(const typename pcl::PointCloud<PointT>::ConstPtr &cloud) {
	cv::Mat img;
	if (cloud->isOrganized()) {
		img = cv::Mat(cloud->height, cloud->width, CV_32F);
		if (!cloud->empty()) {
			for (int h = 0; h < img.rows; h++) {
				for (int w = 0; w < img.cols; w++) {
					img.at<float>(h,w) = cloud->at(w,h).z;
				}
			}
		}
	}
	return img;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr organizePointCloudToImageView(const typename pcl::PointCloud<PointT>::ConstPtr &cloud_in, const cv::Size &im_size, const cv::Mat &K, const cv::Mat &d) {

	// PointT pt;
	// pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
	// typename pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>(im_size.width,im_size.height, pt));

	typename pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>(im_size.width,im_size.height));

	// cloud_out->getMatrixXfMap().topRows(3) = Eigen::MatrixXf::Constant(3, im_size.width*im_size.height, std::numeric_limits<float>::quiet_NaN());

	typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	std::vector<int> tmp;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud, tmp);

	if (cloud->empty()) {
		return cloud_out;
	}

	Eigen::MatrixXf pts3d_e = cloud->getMatrixXfMap().topRows(3).transpose();
	Eigen::Matrix3f R_id_e = Eigen::Matrix3f::Identity();
	Eigen::Vector3f t_id_e = Eigen::Vector3f::Zero();
	cv::Mat pts2d, pts3d, rvec, R_id, t_id;

	cv::eigen2cv(pts3d_e, pts3d);
	cv::eigen2cv(R_id_e, R_id);
	cv::eigen2cv(t_id_e, t_id);

	cv::Rodrigues(R_id, rvec);
	cv::projectPoints(pts3d, rvec, t_id, K, d, pts2d);

	for (int i = 0; i < pts2d.rows; ++i) {
		int col = std::round(pts2d.at<cv::Vec2f>(i,0)[0]);
		int row = std::round(pts2d.at<cv::Vec2f>(i,0)[1]);
		if (col >= 0 && col < cloud_out->width && row >= 0 && row < cloud_out->height) {
			cloud_out->at(col,row) = cloud->points[i];
		}
	}

	return cloud_out;
}

template<typename PointInT, typename PointOutT>
typename pcl::PointCloud<PointOutT>::Ptr colorOrganizedPointCloudFromImageView(const typename pcl::PointCloud<PointInT>::ConstPtr &cloud_in, const cv::Mat &img) {

	typename pcl::PointCloud<PointOutT>::Ptr cloud_out(new pcl::PointCloud<PointOutT>);
	pcl::copyPointCloud<PointInT,PointOutT>(*cloud_in, *cloud_out);

	for (int row = 0; row < img.rows; ++row) {
		for (int col = 0; col < img.cols; ++col) {
			cloud_out->at(col,row).r = img.at<cv::Vec3b>(row,col)[2];
			cloud_out->at(col,row).g = img.at<cv::Vec3b>(row,col)[1];
			cloud_out->at(col,row).b = img.at<cv::Vec3b>(row,col)[0];
		}
	}
	return cloud_out;
}

template<typename PointInT, typename PointOutT>
typename pcl::PointCloud<PointOutT>::Ptr colorPointCloudFromImageView(const typename pcl::PointCloud<PointInT>::ConstPtr &cloud_in, const cv::Mat &img, const cv::Mat &K, const cv::Mat &d) {

	typename pcl::PointCloud<PointOutT>::Ptr cloud_out(new pcl::PointCloud<PointOutT>);

	typename pcl::PointCloud<PointOutT>::Ptr cloud(new pcl::PointCloud<PointOutT>);
	pcl::copyPointCloud<PointInT,PointOutT>(*cloud_in, *cloud);

	std::vector<int> tmp;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, tmp);

	if (cloud->empty()) {
		return cloud_out;
	}

	Eigen::MatrixXf pts3d_e = cloud->getMatrixXfMap().topRows(3).transpose();
	Eigen::Matrix3f R_id_e = Eigen::Matrix3f::Identity();
	Eigen::Vector3f t_id_e = Eigen::Vector3f::Zero();
	cv::Mat pts2d, pts3d, rvec, R_id, t_id;

	cv::eigen2cv(pts3d_e, pts3d);
	cv::eigen2cv(R_id_e, R_id);
	cv::eigen2cv(t_id_e, t_id);

	cv::Rodrigues(R_id, rvec);
	cv::projectPoints(pts3d, rvec, t_id, K, d, pts2d);

	PointOutT pt;
	for (int i = 0; i < pts2d.rows; ++i) {
		int col = std::round(pts2d.at<cv::Vec2f>(i,0)[0]);
		int row = std::round(pts2d.at<cv::Vec2f>(i,0)[1]);
		if (col >= 0 && col < img.cols && row >= 0 && row < img.rows) {
			pt = cloud->points[i];
			pt.r = img.at<cv::Vec3b>(row,col)[2];
			pt.g = img.at<cv::Vec3b>(row,col)[1];
			pt.b = img.at<cv::Vec3b>(row,col)[0];
			cloud_out->push_back(pt);
		}
	}

	return cloud_out;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr pointCloudFromDepthImage(const cv::Mat &depth_img, const Eigen::Matrix3f &K, bool org) {
	typename pcl::PointCloud<PointT>::Ptr res;
	if (org) {
		res = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth_img.cols, depth_img.rows));
		for (int row = 0; row < depth_img.rows; ++row) {
			for (int col = 0; col < depth_img.cols; ++col) {
				float d = (float)(depth_img.at<unsigned short>(row,col));
				PointT pt;
				pt.x = (col - K(0,2)) * d / K(0,0);
				pt.y = (row - K(1,2)) * d / K(1,1);
				pt.z = d;
				res->at(col,row) = pt;
			}
		}
	} else {
		res = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
		res->resize(depth_img.cols*depth_img.rows);
		int num = 0;
		for (int row = 0; row < depth_img.rows; ++row) {
			for (int col = 0; col < depth_img.cols; ++col) {
				float d = (float)(depth_img.at<unsigned short>(row,col));
				if (d > 0.0) {
					PointT pt;
					pt.x = (col - K(0,2)) * d / K(0,0);
					pt.y = (row - K(1,2)) * d / K(1,1);
					pt.z = d;
					res->at(num++) = pt;
				}
			}
		}
		res->resize(num);
	}
	return res;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr pointCloudFromRGBDImages(const cv::Mat &rgb_img, const cv::Mat &depth_img, const Eigen::Matrix3f &K, bool org) {
	typename pcl::PointCloud<PointT>::Ptr res;
	if (org) {
		res = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(depth_img.cols, depth_img.rows));
		for (int row = 0; row < depth_img.rows; ++row) {
			for (int col = 0; col < depth_img.cols; ++col) {
				float d = (float)(depth_img.at<unsigned short>(row,col));
				cv::Vec3b c = rgb_img.at<cv::Vec3b>(row,col);
				PointT pt;
				pt.x = (col - K(0,2)) * d / K(0,0);
				pt.y = (row - K(1,2)) * d / K(1,1);
				pt.z = d;
				pt.r = c[2];
				pt.g = c[1];
				pt.b = c[0];
				res->at(col,row) = pt;
			}
		}
	} else {
		res = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
		res->resize(depth_img.cols*depth_img.rows);
		int num = 0;
		for (int row = 0; row < depth_img.rows; ++row) {
			for (int col = 0; col < depth_img.cols; ++col) {
				float d = (float)(depth_img.at<unsigned short>(row,col));
				cv::Vec3b c = rgb_img.at<cv::Vec3b>(row,col);
				if (d > 0.0) {
					PointT pt;
					pt.x = (col - K(0,2)) * d / K(0,0);
					pt.y = (row - K(1,2)) * d / K(1,1);
					pt.z = d;
					pt.r = c[2];
					pt.g = c[1];
					pt.b = c[0];
					res->at(num++) = pt;
				}
			}
		}
		res->resize(num);
	}
	return res;
}

#endif /* SISYPHUS_REGISTRATION_UTILITIES_HPP */
