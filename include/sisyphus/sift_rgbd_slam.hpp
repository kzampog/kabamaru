#ifndef SISYPHUS_SIFT_RGBD_SLAM_HPP
#define SISYPHUS_SIFT_RGBD_SLAM_HPP

#include <queue>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sisyphus/sift_engine.hpp>

class SLAM {

private:
	SIFTEngine sift_engine;

	bool use_sift;
	bool fail_on_sift_failure;
	bool use_3d_to_3d_matches;
	float max_transfer_error;
	float max_reproj_error;
	int min_inlier_count;
	int ransac_iter;
	bool use_icp;
	int icp_iter;
	float icp_dist_thresh;

	float model_res;
	float normal_radius;
	float mls_radius;
	bool mls_poly_fit;

public:
	struct View {
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloud;
		cv::Mat imageRGB;
		cv::Mat intrinsicsMatrix;
		cv::Mat distortionCoefficients;
		Eigen::MatrixXf keypointImageCoordinates;
		Eigen::MatrixXf keypointWorldCoordinates;
		std::vector<float> keypointDescriptors;
		Eigen::Matrix4f initialPose;
		Eigen::Matrix4f pose;
		std::vector<Eigen::Vector4f> frustum;
	};

	struct sceneModel {
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloud;
		Eigen::MatrixXf keypointWorldCoordinates;
		std::vector<float> keypointDescriptors;
		std::vector<View> registeredViews;
	};

	std::queue<View> unregisteredViews;
	sceneModel model;

	SLAM();
	// SLAM(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_ptr);
	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	void setDefaultParameters();
	void reset();

	void setEstimateWithSIFTMatching(bool sift_flag);
	void setFailOnSIFTMatchingFailure(bool sift_failure_flag);
	void setUse3DTo3DMatches(bool sift_3d_to_3d_flag);
	void setMaxTransferError(float max_trans_err);
	void setMaxReprojectionError(float max_repr_err);
	void setMinRANSACInlierCount(int min_inliers);
	void setMaxRANSACIterations(int n_iter);
	void setRefineWithICP(bool icp_flag);
	void setMaxICPIterations(int n_iter);
	void setICPDistanceThreshold(float thresh);
	void setModelResolution(float res);
	void setNormalRadius(float rad);
	void setMLSRadius(float rad);
	void setMLSPolynomialFit(bool mls_poly);

	bool getEstimateWithSIFTMatching();
	bool getFailOnSIFTMatchingFailure();
	bool getUse3DTo3DMatches();
	float getMaxTransferError();
	float getMaxReprojectionError();
	int getMinRANSACInlierCount();
	int getMaxRANSACIterations();
	bool getRefineWithICP();
	int getMaxICPIterations();
	float getICPDistanceThreshold();
	float getModelResolution();
	float getNormalRadius();
	float getMLSRadius();
	bool getMLSPolynomialFit();

	View createView(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, const Eigen::Matrix4f &pose = Eigen::Matrix4f::Identity());
	View createView(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, const cv::Mat &K, const cv::Mat &d, const Eigen::Matrix4f &pose = Eigen::Matrix4f::Identity());
	View createView(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud, const cv::Mat &img, const cv::Mat &K, const cv::Mat &d, const Eigen::Matrix4f &pose = Eigen::Matrix4f::Identity());
	View createView(const cv::Mat &img, const cv::Mat &K, const cv::Mat &d, const Eigen::Matrix4f &pose = Eigen::Matrix4f::Identity());

	int estimateViewPose(const View &v, Eigen::Matrix4f& tform);

	void enqueueView(const View &v);
	void clearUnregisteredViews();

	void initializeModelFromView(const View &v);
	int initializeModelFromNextQueuedView();
	int integrateView(const View &v);
	int integrateView(const View &v, Eigen::Matrix4f &pose);
	int integrateNextQueuedView();
	int integrateAllQueuedViews();

	int rigidlyAlignModelToInitialPoses();

	void smoothModelPointCloudMLS();
	
	void clearModel();

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getModelPointCloud();
	void getSIFTModel(Eigen::MatrixXf& loc, std::vector<float>& desrc);
	std::vector<View> getModelRegisteredViews();
	sceneModel getSceneModel();

	std::vector<cv::Mat> getRegisteredViewRGBImages();
	std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> getRegisteredViewPointClouds();
	std::vector<Eigen::Matrix4f> getRegisteredViewPoses();
	std::vector<std::vector<Eigen::Vector4f> > getRegisteredViewFrustums();

	void readSceneModel(const std::string &dir_name);
	void writeSceneModel(const std::string &dir_name);
};

#endif /* SISYPHUS_SIFT_RGBD_SLAM_HPP */
