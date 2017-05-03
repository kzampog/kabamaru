#include <kabamaru/sift_rgbd_slam.hpp>
#include <kabamaru/pointcloud_utilities.hpp>
#include <kabamaru/io_utilities.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>

SLAM::SLAM() {
	setDefaultParameters();
	reset();
}

// SLAM::SLAM(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_ptr) {
// 	setDefaultParameters();
// 	viewer = viewer_ptr;
// }

void SLAM::setDefaultParameters() {
	use_sift = true;
	fail_on_sift_failure = true;
	use_3d_to_3d_matches = true;
	max_transfer_error = 0.001;
	max_reproj_error = 8.0;
	min_inlier_count = 80;
	ransac_iter = 5000;
	use_icp = true;
	icp_iter = 15;
	icp_dist_thresh = 0.01;
	model_res = 0.005;
	normal_radius = 0.03;
	mls_radius = 0.015;
	mls_poly_fit = false;
}

void SLAM::reset() {
	setDefaultParameters();
	clearUnregisteredViews();
	clearModel();
}

void SLAM::setEstimateWithSIFTMatching(bool sift_flag) {
	use_sift = sift_flag;
}

void SLAM::setFailOnSIFTMatchingFailure(bool sift_failure_flag) {
	fail_on_sift_failure = sift_failure_flag;
}

void SLAM::setUse3DTo3DMatches(bool sift_3d_to_3d_flag) {
	use_3d_to_3d_matches = sift_3d_to_3d_flag;
}

void SLAM::setMaxTransferError(float max_trans_err) {
	max_transfer_error = max_trans_err;
}

void SLAM::setMaxReprojectionError(float max_repr_err) {
	max_reproj_error = max_repr_err;
}

void SLAM::setMinRANSACInlierCount(int min_inliers) {
	min_inlier_count = min_inliers;
}

void SLAM::setMaxRANSACIterations(int n_iter) {
	ransac_iter = n_iter;
}

void SLAM::setRefineWithICP(bool icp_flag) {
	use_icp = icp_flag;
}

void SLAM::setMaxICPIterations(int n_iter) {
	icp_iter = n_iter;
}

void SLAM::setICPDistanceThreshold(float thresh) {
	icp_dist_thresh = thresh;
}

void SLAM::setModelResolution(float res) {
	model_res = res;
}

void SLAM::setNormalEstimationRadius(float rad) {
	normal_radius = rad;
}

void SLAM::setMLSRadius(float rad) {
	mls_radius = rad;
}

void SLAM::setMLSPolynomialFit(bool mls_poly) {
	mls_poly_fit = mls_poly;
}

bool SLAM::getEstimateWithSIFTMatching() {
	return use_sift;
}

bool SLAM::getFailOnSIFTMatchingFailure() {
	return fail_on_sift_failure;
}

bool SLAM::getUse3DTo3DMatches() {
	return use_3d_to_3d_matches;
}

float SLAM::getMaxTransferError() {
	return max_transfer_error;
}

float SLAM::getMaxReprojectionError() {
	return max_reproj_error;
}

int SLAM::getMinRANSACInlierCount() {
	return min_inlier_count;
}

int SLAM::getMaxRANSACIterations() {
	return ransac_iter;
}

bool SLAM::getRefineWithICP() {
	return use_icp;
}

int SLAM::getMaxICPIterations() {
	return icp_iter;
}

float SLAM::getICPDistanceThreshold() {
	return icp_dist_thresh;
}

float SLAM::getModelResolution() {
	return model_res;
}

float SLAM::getNormalEstimationRadius() {
	return normal_radius;
}

float SLAM::getMLSRadius() {
	return mls_radius;
}

bool SLAM::getMLSPolynomialFit() {
	return mls_poly_fit;
}

SLAM::View SLAM::createView(const cv::Mat &rgb_img, const cv::Mat &depth_img, const Eigen::Matrix3f &K, const Eigen::Matrix4f &pose) {
	View v;
	rgb_img.copyTo(v.imageRGB);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = pointCloudFromRGBDImages<pcl::PointXYZRGB>(rgb_img, depth_img, K, true);
	cv::Mat K_cv;
	cv::eigen2cv(K, K_cv);
	K_cv.copyTo(v.intrinsicsMatrix);

	std::vector<SiftGPU::SiftKeypoint> keys;
	sift_engine.detectFeatures(v.imageRGB, keys, v.keypointDescriptors);
	v.keypointImageCoordinates = extractSIFTKeypoint2DCoordinates(keys);
	extractKeypoint3DCoordinatesFromOrganizedPointCloud<pcl::PointXYZRGB>(cloud, v.keypointImageCoordinates, v.keypointDescriptors, v.keypointWorldCoordinates);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_d = downSamplePointCloud<pcl::PointXYZRGB>(cloud, model_res);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_f = estimatePointCloudNormals<pcl::PointXYZRGB,pcl::PointXYZRGBNormal>(cloud_d, normal_radius);

	std::vector<int> tmp;
	pcl::removeNaNFromPointCloud(*cloud_f, *cloud_f, tmp);
	pcl::removeNaNNormalsFromPointCloud(*cloud_f, *cloud_f, tmp);
	v.pointCloud = cloud_f;
	v.frustum = viewConeFromPointCloud<pcl::PointXYZRGBNormal>(cloud_f);

	v.initialPose = pose;
	v.pose = pose;
	Eigen::Matrix3f R = pose.block<3,3>(0,0);
	Eigen::Vector3f t = pose.block<3,1>(0,3);
	transformConvexPolytope(v.frustum, pose);
	v.keypointWorldCoordinates = (R*v.keypointWorldCoordinates).colwise() + t;
	pcl::transformPointCloudWithNormals(*v.pointCloud, *v.pointCloud, pose);

	return v;
}

SLAM::View SLAM::createView(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, const Eigen::Matrix4f &pose) {
	View v;
	v.imageRGB = organizedPointCloudToRGBImage<pcl::PointXYZRGB>(cloud);
	// v.imageDepth = organizedPointCloudToDepthImage<pcl::PointXYZRGB>(cloud);

	std::vector<SiftGPU::SiftKeypoint> keys;
	sift_engine.detectFeatures(v.imageRGB, keys, v.keypointDescriptors);
	v.keypointImageCoordinates = extractSIFTKeypoint2DCoordinates(keys);
	extractKeypoint3DCoordinatesFromOrganizedPointCloud<pcl::PointXYZRGB>(cloud, v.keypointImageCoordinates, v.keypointDescriptors, v.keypointWorldCoordinates);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_d = downSamplePointCloud<pcl::PointXYZRGB>(cloud, model_res);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_f = estimatePointCloudNormals<pcl::PointXYZRGB,pcl::PointXYZRGBNormal>(cloud_d, normal_radius);

	std::vector<int> tmp;
	pcl::removeNaNFromPointCloud(*cloud_f, *cloud_f, tmp);
	pcl::removeNaNNormalsFromPointCloud(*cloud_f, *cloud_f, tmp);
	v.pointCloud = cloud_f;
	v.frustum = viewConeFromPointCloud<pcl::PointXYZRGBNormal>(cloud_f);

	v.initialPose = pose;
	v.pose = pose;
	Eigen::Matrix3f R = pose.block<3,3>(0,0);
	Eigen::Vector3f t = pose.block<3,1>(0,3);
	transformConvexPolytope(v.frustum, pose);
	v.keypointWorldCoordinates = (R*v.keypointWorldCoordinates).colwise() + t;
	pcl::transformPointCloudWithNormals(*v.pointCloud, *v.pointCloud, pose);

	return v;
}

SLAM::View SLAM::createView(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud, const cv::Mat &K, const cv::Mat &d, const Eigen::Matrix4f &pose) {
	View v = createView(cloud, pose);
	K.copyTo(v.intrinsicsMatrix);
	d.copyTo(v.distortionCoefficients);
	return v;
}

SLAM::View SLAM::createView(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud, const cv::Mat &img, const cv::Mat &K, const cv::Mat &d, const Eigen::Matrix4f &pose) {
	View v;
	img.copyTo(v.imageRGB);
	K.copyTo(v.intrinsicsMatrix);
	d.copyTo(v.distortionCoefficients);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_reg = organizePointCloudToImageView<pcl::PointXYZ>(cloud, img.size(), K, d);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_col = colorPointCloudFromImageView<pcl::PointXYZ,pcl::PointXYZRGB>(cloud, img, K, d);

	std::vector<SiftGPU::SiftKeypoint> keys;
	sift_engine.detectFeatures(v.imageRGB, keys, v.keypointDescriptors);
	v.keypointImageCoordinates = extractSIFTKeypoint2DCoordinates(keys);
	extractKeypoint3DCoordinatesFromOrganizedPointCloud<pcl::PointXYZ>(cloud_reg, v.keypointImageCoordinates, v.keypointDescriptors, v.keypointWorldCoordinates);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_d = downSamplePointCloud<pcl::PointXYZRGB>(cloud_col, model_res);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_f = estimatePointCloudNormals<pcl::PointXYZRGB,pcl::PointXYZRGBNormal>(cloud_d, normal_radius);

	std::vector<int> tmp;
	pcl::removeNaNFromPointCloud(*cloud_f, *cloud_f, tmp);
	pcl::removeNaNNormalsFromPointCloud(*cloud_f, *cloud_f, tmp);
	v.pointCloud = cloud_f;
	v.frustum = viewConeFromPointCloud<pcl::PointXYZRGBNormal>(cloud_f);

	v.initialPose = pose;
	v.pose = pose;
	Eigen::Matrix3f R = pose.block<3,3>(0,0);
	Eigen::Vector3f t = pose.block<3,1>(0,3);
	transformConvexPolytope(v.frustum, pose);
	v.keypointWorldCoordinates = (R*v.keypointWorldCoordinates).colwise() + t;
	pcl::transformPointCloudWithNormals(*v.pointCloud, *v.pointCloud, pose);

	return v;
}

SLAM::View SLAM::createView(const cv::Mat &img, const cv::Mat &K, const cv::Mat &d, const Eigen::Matrix4f &pose) {
	View v;
	img.copyTo(v.imageRGB);
	K.copyTo(v.intrinsicsMatrix);
	d.copyTo(v.distortionCoefficients);
	std::vector<SiftGPU::SiftKeypoint> keys;
	sift_engine.detectFeatures(v.imageRGB, keys, v.keypointDescriptors);
	v.keypointImageCoordinates = extractSIFTKeypoint2DCoordinates(keys);
	v.initialPose = pose;
	v.pose = pose;

	return v;
}

int SLAM::estimateViewPose(const View &v, Eigen::Matrix4f& pose) {
	Eigen::Matrix4f tform_est = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f tform_sift;
	int ransac_success;
	if (use_sift) {
		std::vector<int> inliers;
		if (use_3d_to_3d_matches) {
			ransac_success = estimateRigidTransformSIFT3DTo3DRANSAC(v.keypointWorldCoordinates, v.keypointDescriptors, model.keypointWorldCoordinates, model.keypointDescriptors, &sift_engine, ransac_iter, max_transfer_error, min_inlier_count, tform_sift, inliers);
		} else {
			ransac_success = estimateRigidTransformSIFT2DTo3DRANSAC(v.keypointImageCoordinates, v.keypointDescriptors, model.keypointWorldCoordinates, model.keypointDescriptors, v.intrinsicsMatrix, v.distortionCoefficients, &sift_engine, ransac_iter, max_reproj_error, min_inlier_count, tform_sift, inliers);
		}
		if (ransac_success > 0) {
			tform_est = tform_sift;
		} else {
			pose = v.pose;
			if (fail_on_sift_failure) return 0;
		}
	}

	if (!use_icp) {
		pose = tform_est * v.pose;
		return 1;
	}

// std::cout << tform_est << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_t(new pcl::PointCloud<pcl::PointXYZRGBNormal>);	
	pcl::transformPointCloudWithNormals(*v.pointCloud, *cloud_t, tform_est);
	std::vector<Eigen::Vector4f> frustum_tmp = v.frustum;
	transformConvexPolytope(frustum_tmp, tform_est);

	std::vector<std::vector<Eigen::Vector4f> > frustum_intersections;
	for (int i = 0; i < model.registeredViews.size(); i++) {
		std::vector<Eigen::Vector4f> tmp = model.registeredViews[i].frustum;
		tmp.insert(tmp.end(), frustum_tmp.begin(), frustum_tmp.end());
		frustum_intersections.push_back(tmp);
	}

	std::vector<int> pt_ind_frame, pt_ind_model;
	pointIndicesInConvexPolytopeUnion<pcl::PointXYZRGBNormal>(cloud_t, frustum_intersections, pt_ind_frame);
	pointIndicesInConvexPolytopeUnion<pcl::PointXYZRGBNormal>(model.pointCloud, frustum_intersections, pt_ind_model);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_clipped = extractPointCloudFromIndices<pcl::PointXYZRGBNormal>(cloud_t, pt_ind_frame, false, false);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_pc_clipped = extractPointCloudFromIndices<pcl::PointXYZRGBNormal>(model.pointCloud, pt_ind_model, false, false);

// viewer->removeAllPointClouds();
// viewer->addPointCloud<pcl::PointXYZRGBNormal>(cloud_t, "cloud1");
// viewer->addPointCloud<pcl::PointXYZRGBNormal>(model.pointCloud, "cloud2");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "cloud1");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "cloud2");
// viewer->spin();

// viewer->removeAllPointClouds();
// viewer->addPointCloud<pcl::PointXYZRGBNormal>(cloud_clipped, "cloud1");
// viewer->addPointCloud<pcl::PointXYZRGBNormal>(model_pc_clipped, "cloud2");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "cloud1");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "cloud2");
// viewer->spin();

	Eigen::Matrix4f tform_icp = Eigen::Matrix4f::Identity();
	int icp_success = refineTransformICP<pcl::PointXYZRGBNormal>(cloud_clipped, model_pc_clipped, tform_icp, icp_iter, icp_dist_thresh);

	pose = tform_icp * tform_est * v.pose;
// pcl::transformPointCloudWithNormals(*cloud_clipped, *cloud_clipped, tform_icp);
// viewer->updatePointCloud<pcl::PointXYZRGBNormal>(cloud_clipped, "cloud1");
// viewer->spin();

	return icp_success;
}

void SLAM::enqueueView(const View &v) {
	unregisteredViews.push(v);
}

void SLAM::clearUnregisteredViews(){
	unregisteredViews = std::queue<View>();
}

void SLAM::initializeModelFromView(const View& v) {
	clearModel();
	*model.pointCloud = *v.pointCloud;
	model.keypointWorldCoordinates = v.keypointWorldCoordinates;
	model.keypointDescriptors = v.keypointDescriptors;
	model.registeredViews.push_back(v);
}

int SLAM::initializeModelFromNextQueuedView() {
	if (unregisteredViews.empty()) {
		return 0;
	}
	View v = unregisteredViews.front();
	initializeModelFromView(v);
	unregisteredViews.pop();
	return 1;
}

int SLAM::integrateView(const View& v_in) {
	Eigen::Matrix4f pose;
	return integrateView(v_in, pose);
}

int SLAM::integrateView(const View &v_in, Eigen::Matrix4f &pose) {
	if (model.keypointWorldCoordinates.cols() == 0) {
		pose = v_in.pose;
		initializeModelFromView(v_in);
		return 1;
	}

	View v = v_in;

	Eigen::Matrix4f pose_init = v.pose;
	int success = estimateViewPose(v, pose);

	if (success == 0) {
		return 0;
	}

	v.pose = pose;
	Eigen::Matrix4f tform = v.pose * pose_init.inverse();
	Eigen::Matrix3f R = tform.block<3,3>(0,0);
	Eigen::Vector3f t = tform.block<3,1>(0,3);
	transformConvexPolytope(v.frustum, tform);
	v.keypointWorldCoordinates = (R*v.keypointWorldCoordinates).colwise() + t;
	pcl::transformPointCloudWithNormals(*v.pointCloud, *v.pointCloud, tform);
	model.registeredViews.push_back(v);

	Eigen::MatrixXf tmp = Eigen::MatrixXf(model.keypointWorldCoordinates.rows(), model.keypointWorldCoordinates.cols()+v.keypointWorldCoordinates.cols());
	tmp << model.keypointWorldCoordinates, v.keypointWorldCoordinates;
	model.keypointWorldCoordinates = tmp;
	model.keypointDescriptors.insert(model.keypointDescriptors.end(), v.keypointDescriptors.begin(), v.keypointDescriptors.end());

	*model.pointCloud += *v.pointCloud;
	model.pointCloud = downSamplePointCloud<pcl::PointXYZRGBNormal>(model.pointCloud, model_res);

	return 1;
}

int SLAM::integrateNextQueuedView() {
	if (unregisteredViews.empty()) {
		return 0;
	}
	View v = unregisteredViews.front();
	unregisteredViews.pop();
	int success = integrateView(v);
	if (!success) {
		unregisteredViews.push(v);
	}
	return success;
}

int SLAM::integrateAllQueuedViews() {
	int num_pending, success, integrated_curr, num_integrated = 0;
	while (true) {
		integrated_curr = 0;
		num_pending = unregisteredViews.size();
		for (int i = 0; i < num_pending; i++) {
			success = integrateNextQueuedView();
			integrated_curr += success;
			num_integrated += success;
		}
		if (integrated_curr == 0) {
			break;
		}
	}
	return num_integrated;
}

int SLAM::rigidlyAlignModelToInitialPoses() {
	std::vector<Eigen::Matrix4f> src_poses(model.registeredViews.size()), dst_poses(model.registeredViews.size());
	for (int i = 0; i < model.registeredViews.size(); ++i) {
		src_poses[i] = model.registeredViews[i].pose;
		dst_poses[i] = model.registeredViews[i].initialPose;
	}
	Eigen::Matrix4f tform;
	if (!rigidlyAlignPoseSets(src_poses, dst_poses, tform)) return 0;
	Eigen::Matrix3f R = tform.block<3,3>(0,0);
	Eigen::Vector3f t = tform.block<3,1>(0,3);

	transformPointCloudWithNormals(*model.pointCloud, *model.pointCloud, tform);
	model.keypointWorldCoordinates = (R*model.keypointWorldCoordinates).colwise() + t;

	for (int i = 0; i < model.registeredViews.size(); ++i) {
		if (model.registeredViews[i].pointCloud && (!model.registeredViews[i].pointCloud->empty())) {
			transformPointCloudWithNormals(*model.registeredViews[i].pointCloud, *model.registeredViews[i].pointCloud, tform);
		}
		if (model.registeredViews[i].keypointWorldCoordinates.cols() > 0) {
			model.registeredViews[i].keypointWorldCoordinates = (R*model.registeredViews[i].keypointWorldCoordinates).colwise() + t;
		}
		model.registeredViews[i].pose = tform * model.registeredViews[i].pose;
		transformConvexPolytope(model.registeredViews[i].frustum, tform);
	}

	return 1;
}

void SLAM::smoothModelPointCloudMLS() {
	model.pointCloud = smoothPointCloudMLS<pcl::PointXYZRGBNormal,pcl::PointXYZRGBNormal>(model.pointCloud, mls_radius, mls_poly_fit);
	model.pointCloud = downSamplePointCloud<pcl::PointXYZRGBNormal>(model.pointCloud, model_res);
}

void SLAM::clearModel() {
	model.pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	model.keypointWorldCoordinates.resize(3,0);
	model.keypointDescriptors.clear();
	model.registeredViews.clear();
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr SLAM::getModelPointCloud() {
	return model.pointCloud;
}

void SLAM::getSIFTModel(Eigen::MatrixXf& loc, std::vector<float>& descr) {
	loc = model.keypointWorldCoordinates;
	descr = model.keypointDescriptors;
}

std::vector<SLAM::View> SLAM::getModelRegisteredViews() {
	return model.registeredViews;
}

SLAM::sceneModel SLAM::getSceneModel() {
	return model;
}


std::vector<cv::Mat> SLAM::getRegisteredViewRGBImages() {
	std::vector<cv::Mat> res(model.registeredViews.size());
	for (int i = 0; i < res.size(); ++i) {
		res[i] = model.registeredViews[i].imageRGB;
	}
	return res;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> SLAM::getRegisteredViewPointClouds() {
	std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> res(model.registeredViews.size());
	for (int i = 0; i < res.size(); ++i) {
		res[i] = model.registeredViews[i].pointCloud;
	}
	return res;
}

std::vector<Eigen::Matrix4f> SLAM::getRegisteredViewPoses() {
	std::vector<Eigen::Matrix4f> res(model.registeredViews.size());
	for (int i = 0; i < res.size(); ++i) {
		res[i] = model.registeredViews[i].pose;
	}
	return res;
}

std::vector<std::vector<Eigen::Vector4f> > SLAM::getRegisteredViewFrustums() {
	std::vector<std::vector<Eigen::Vector4f> > res;
	for (int i = 0; i < res.size(); ++i) {
		res[i] = model.registeredViews[i].frustum;
	}
	return res;
}

void SLAM::readSceneModel(const std::string &dir_name) {
	boost::filesystem::path model_dir(dir_name);

	// Point cloud
	std::string pcd_fname = (model_dir/boost::filesystem::path("model_point_cloud.pcd")).string();
	if (boost::filesystem::exists(pcd_fname)) {
		model.pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::io::loadPCDFile(pcd_fname, *model.pointCloud);
	}

	// SIFT model
	std::string keypoint_3dloc_fname = (model_dir/boost::filesystem::path("model_keypoint_world_coordinates.dat")).string();
	if (boost::filesystem::exists(keypoint_3dloc_fname)) {
		readEigenMatrixFromFile(keypoint_3dloc_fname, model.keypointWorldCoordinates);
	}
	std::string keypoint_descr_fname = (model_dir/boost::filesystem::path("model_keypoint_descriptors.dat")).string();
	if (boost::filesystem::exists(keypoint_descr_fname)) {
		readVectorFromFile(keypoint_descr_fname, model.keypointDescriptors);
	}

	// Registered Views
	std::string views_dir_name = (model_dir/boost::filesystem::path("registered_views")).string();
	if (!boost::filesystem::exists(views_dir_name)) {
		return;
	}

	// Count registered views
	boost::filesystem::path views_dir(views_dir_name);

	int n_views = 0;
	std::string prefix = "pose";
	for (boost::filesystem::directory_iterator it(views_dir); it != boost::filesystem::directory_iterator(); ++it) {
		std::string name = it->path().filename().string();
		if (std::mismatch(prefix.begin(), prefix.end(), name.begin()).first == prefix.end()) {
			n_views++;
		}
	}

	// Copy data from files
	model.registeredViews = std::vector<View>(n_views);
	for (int i = 0; i < n_views; ++i) {
		std::ostringstream suff;
		suff << std::setw(4) << std::setfill('0') << i;

		std::string img_fname = (views_dir/boost::filesystem::path("image_" + suff.str() + ".png")).string();
		if (boost::filesystem::exists(img_fname)) {
			model.registeredViews[i].imageRGB = cv::imread(img_fname, CV_LOAD_IMAGE_COLOR);
		}

		std::string intrinsics_fname = (views_dir/boost::filesystem::path("intrinsics_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(intrinsics_fname)) {
			Eigen::MatrixXf K;
			readEigenMatrixFromFile(intrinsics_fname, K);
			cv::eigen2cv(K, model.registeredViews[i].intrinsicsMatrix);
		}

		std::string distortion_fname = (views_dir/boost::filesystem::path("distortion_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(distortion_fname)) {
			Eigen::MatrixXf d;
			readEigenMatrixFromFile(distortion_fname, d);
			cv::eigen2cv(d, model.registeredViews[i].distortionCoefficients);
		}

		std::string pcd_fname = (views_dir/boost::filesystem::path("point_cloud_" + suff.str() + ".pcd")).string();
		if (boost::filesystem::exists(pcd_fname)) {
			model.registeredViews[i].pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
			pcl::io::loadPCDFile(pcd_fname, *model.registeredViews[i].pointCloud);
		}

		std::string keypoint_2dloc_fname = (views_dir/boost::filesystem::path("keypoint_image_coordinates_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(keypoint_2dloc_fname)) {
			readEigenMatrixFromFile(keypoint_2dloc_fname, model.registeredViews[i].keypointImageCoordinates);
		}

		std::string keypoint_3dloc_fname = (views_dir/boost::filesystem::path("keypoint_world_coordinates_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(keypoint_3dloc_fname)) {
			readEigenMatrixFromFile(keypoint_3dloc_fname, model.registeredViews[i].keypointWorldCoordinates);
		}

		std::string keypoint_descriptors_fname = (views_dir/boost::filesystem::path("keypoint_descriptors_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(keypoint_descriptors_fname)) {
			readVectorFromFile(keypoint_descriptors_fname, model.registeredViews[i].keypointDescriptors);
		}

		std::string initial_pose_fname = (views_dir/boost::filesystem::path("initial_pose_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(initial_pose_fname)) {
			readEigenMatrixFromFile(initial_pose_fname, model.registeredViews[i].initialPose);
		}

		std::string pose_fname = (views_dir/boost::filesystem::path("pose_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(pose_fname)) {
			readEigenMatrixFromFile(pose_fname, model.registeredViews[i].pose);
		}

		std::string frustum_fname = (views_dir/boost::filesystem::path("frustum_" + suff.str() + ".dat")).string();
		if (boost::filesystem::exists(frustum_fname)) {
			model.registeredViews[i].frustum = std::vector<Eigen::Vector4f>(4);
			Eigen::Matrix4f frustum_mat;
			readEigenMatrixFromFile(frustum_fname, frustum_mat);
			for (int j = 0; j < 4; ++j) {
				model.registeredViews[i].frustum[j] = frustum_mat.col(j);
			}
		}
	}
}

void SLAM::writeSceneModel(const std::string &dir_name) {
	boost::filesystem::path model_dir(dir_name);
	boost::filesystem::create_directories(model_dir);

	// Point cloud
	if (model.pointCloud && (!model.pointCloud->empty())) {
		std::string pcd_fname = (model_dir/boost::filesystem::path("model_point_cloud.pcd")).string();
		pcl::io::savePCDFile(pcd_fname, *model.pointCloud, true);
	}

	// SIFT model
	if (model.keypointWorldCoordinates.cols() > 0) {
		std::string keypoint_3dloc_fname = (model_dir/boost::filesystem::path("model_keypoint_world_coordinates.dat")).string();
		writeEigenMatrixToFile(keypoint_3dloc_fname, model.keypointWorldCoordinates);
	}

	if (!model.keypointDescriptors.empty()) {
		std::string keypoint_descr_fname = (model_dir/boost::filesystem::path("model_keypoint_descriptors.dat")).string();
		writeVectorToFile(keypoint_descr_fname, model.keypointDescriptors);
	}

	// Registered Views
	int n_views = model.registeredViews.size();
	if (n_views == 0) {
		return;
	}

	std::string views_dir_name = (model_dir/boost::filesystem::path("registered_views")).string();
	boost::filesystem::path views_dir(views_dir_name);
	boost::filesystem::create_directories(views_dir);

	for (int i = 0; i < n_views; ++i) {
		std::ostringstream suff;
		suff << std::setw(4) << std::setfill('0') << i;

		if (!model.registeredViews[i].imageRGB.empty()) {
			std::string img_fname = (views_dir/boost::filesystem::path("image_" + suff.str() + ".png")).string();
			cv::imwrite(img_fname, model.registeredViews[i].imageRGB);
		}

		if (!model.registeredViews[i].intrinsicsMatrix.empty()) {
			Eigen::MatrixXf K;
			cv::cv2eigen(model.registeredViews[i].intrinsicsMatrix, K);
			std::string intrinsics_fname = (views_dir/boost::filesystem::path("intrinsics_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(intrinsics_fname, K);
		}

		if (!model.registeredViews[i].distortionCoefficients.empty()) {
			Eigen::MatrixXf d;
			cv::cv2eigen(model.registeredViews[i].distortionCoefficients, d);
			std::string distortion_fname = (views_dir/boost::filesystem::path("distortion_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(distortion_fname, d);
		}

		if (model.registeredViews[i].pointCloud && (!model.registeredViews[i].pointCloud->empty())) {
			std::string pcd_fname = (views_dir/boost::filesystem::path("point_cloud_" + suff.str() + ".pcd")).string();
			pcl::io::savePCDFile(pcd_fname, *model.registeredViews[i].pointCloud, true);
		}

		if (model.registeredViews[i].keypointImageCoordinates.cols() > 0) {
			std::string keypoint_2dloc_fname = (views_dir/boost::filesystem::path("keypoint_image_coordinates_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(keypoint_2dloc_fname, model.registeredViews[i].keypointImageCoordinates);
		}

		if (model.registeredViews[i].keypointWorldCoordinates.cols() > 0) {
			std::string keypoint_3dloc_fname = (views_dir/boost::filesystem::path("keypoint_world_coordinates_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(keypoint_3dloc_fname, model.registeredViews[i].keypointWorldCoordinates);
		}

		if (!model.registeredViews[i].keypointDescriptors.empty()) {
			std::string keypoint_descriptors_fname = (views_dir/boost::filesystem::path("keypoint_descriptors_" + suff.str() + ".dat")).string();
			writeVectorToFile(keypoint_descriptors_fname, model.registeredViews[i].keypointDescriptors);
		}

		if (true) {
			std::string initial_pose_fname = (views_dir/boost::filesystem::path("initial_pose_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(initial_pose_fname, model.registeredViews[i].initialPose);
		}

		if (true) {
			std::string pose_fname = (views_dir/boost::filesystem::path("pose_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(pose_fname, model.registeredViews[i].pose);
		}

		if (model.registeredViews[i].frustum.size() == 4) {
			Eigen::Matrix4f frustum_mat;
			for (int j = 0; j < 4; ++j) {
				frustum_mat.col(j) = model.registeredViews[i].frustum[j];
			}
			std::string frustum_fname = (views_dir/boost::filesystem::path("frustum_" + suff.str() + ".dat")).string();
			writeEigenMatrixToFile(frustum_fname, frustum_mat);
		}
	}
}
