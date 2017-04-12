#ifndef POINTCLOUD_UTILITIES_HPP
#define POINTCLOUD_UTILITIES_HPP

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/impl/plane_clipper3D.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/surface/mls.h>

#include <sisyphus/registration_utilities.hpp>

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr extractPointCloudFromIndices(const typename pcl::PointCloud<PointT>::ConstPtr &cloud_in, const std::vector<int> &inlier_ind, bool neg, bool keep_org) {
	typename pcl::PointCloud<PointT>::Ptr cloud_out (new pcl::PointCloud<PointT>);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	inliers->indices = inlier_ind;
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(cloud_in);
	extract.setIndices(inliers);
	extract.setKeepOrganized(keep_org);
	extract.setNegative(neg);
	extract.filter(*cloud_out);
	return cloud_out;
}

template<typename PointT>
void pointIndicesInConvexPolytope(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<Eigen::Vector4f> &planes, std::vector<int>& inlier_ind) {
	if (planes.size() == 0) return;

	std::vector<int> ind_intersection;
	for (int i = 0; i < planes.size(); i++) {
		std::vector<int> tmp_in, tmp;
		pcl::PlaneClipper3D<PointT> clipper(planes[i]);
		clipper.clipPointCloud3D(*cloud, tmp_in, tmp);
		if (i == 0) ind_intersection = tmp_in;
		std::vector<int> tmp_intersection = ind_intersection;
		ind_intersection.clear();
		std::sort(tmp_intersection.begin(), tmp_intersection.end());
		std::sort(tmp_in.begin(), tmp_in.end());
		std::set_intersection(tmp_intersection.begin(), tmp_intersection.end(), tmp_in.begin(), tmp_in.end(), std::back_inserter(ind_intersection));
	}
	inlier_ind = ind_intersection;
}

template<typename PointT>
void pointIndicesInConvexPolytopeUnion(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<std::vector<Eigen::Vector4f> > &polytopes, std::vector<int>& inlier_ind) {

	std::vector<int> ind_union;
	for (int i = 0; i < polytopes.size(); i++) {
		std::vector<int> inlier_tmp, tmp_union;

		pointIndicesInConvexPolytope<PointT>(cloud, polytopes[i], inlier_tmp);

		tmp_union = ind_union;
		ind_union.clear();
		std::sort(inlier_tmp.begin(), inlier_tmp.end());
		std::sort(tmp_union.begin(), tmp_union.end());
		std::set_union(tmp_union.begin(), tmp_union.end(), inlier_tmp.begin(), inlier_tmp.end(), std::back_inserter(ind_union));
	}
	inlier_ind = ind_union;
}

template<typename PointT>
std::vector<Eigen::Vector4f> viewConeFromPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud) {
	std::vector<Eigen::Vector4f> planes(4);
	if (cloud->empty()) {
		return planes;
	}

	int min_x_ind, max_x_ind, min_y_ind, max_y_ind;
	Eigen::MatrixXf pts = cloud->getMatrixXfMap();
	pts.row(0).minCoeff(&min_x_ind);
	pts.row(0).maxCoeff(&max_x_ind);
	pts.row(1).minCoeff(&min_y_ind);
	pts.row(1).maxCoeff(&max_y_ind);

	Eigen::Vector3f min_x_pt(pts(0,min_x_ind),pts(1,min_x_ind),pts(2,min_x_ind));
	Eigen::Vector3f max_x_pt(pts(0,max_x_ind),pts(1,max_x_ind),pts(2,max_x_ind));
	Eigen::Vector3f min_y_pt(pts(0,min_y_ind),pts(1,min_y_ind),pts(2,min_y_ind));
	Eigen::Vector3f max_y_pt(pts(0,max_y_ind),pts(1,max_y_ind),pts(2,max_y_ind));

	Eigen::Vector3f left = Eigen::Vector3f(0,1,0).cross(min_x_pt.normalized());
	Eigen::Vector3f right = max_x_pt.normalized().cross(Eigen::Vector3f(0,1,0));
	Eigen::Vector3f top = min_y_pt.normalized().cross(Eigen::Vector3f(1,0,0));
	Eigen::Vector3f bottom = Eigen::Vector3f(1,0,0).cross(max_y_pt.normalized());

	planes[0] = Eigen::Vector4f(left(0),left(1),left(2),0);
	planes[1] = Eigen::Vector4f(right(0),right(1),right(2),0);
	planes[2] = Eigen::Vector4f(top(0),top(1),top(2),0);
	planes[3] = Eigen::Vector4f(bottom(0),bottom(1),bottom(2),0);

	return planes;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr coordinateRangeClipPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud_in, bool keep_org, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax) {
	typename pcl::ConditionAnd<PointT>::Ptr cond(new pcl::ConditionAnd<PointT>());
	cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr (new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GE, xMin)));
	cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr (new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LE, xMax)));
	cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr (new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GE, yMin)));
	cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr (new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LE, yMax)));
	cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr (new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GE, zMin)));
	cond->addComparison(typename pcl::FieldComparison<PointT>::Ptr (new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LE, zMax)));

	typename pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>);
	pcl::ConditionalRemoval<PointT> condrem;
	condrem.setCondition(cond);
	condrem.setInputCloud(cloud_in);
	condrem.setKeepOrganized(keep_org);
	condrem.filter(*cloud_out);

	return cloud_out;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr colorRangeClipPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud_in, bool keep_org, int rMin, int rMax, int gMin, int gMax, int bMin, int bMax) {
	typename pcl::ConditionAnd<PointT>::Ptr cond(new pcl::ConditionAnd<PointT>());
	cond->addComparison(typename pcl::PackedRGBComparison<PointT>::Ptr (new pcl::PackedRGBComparison<PointT> ("r", pcl::ComparisonOps::GE, rMin)));
	cond->addComparison(typename pcl::PackedRGBComparison<PointT>::Ptr (new pcl::PackedRGBComparison<PointT> ("r", pcl::ComparisonOps::LE, rMax)));
	cond->addComparison(typename pcl::PackedRGBComparison<PointT>::Ptr (new pcl::PackedRGBComparison<PointT> ("g", pcl::ComparisonOps::GE, gMin)));
	cond->addComparison(typename pcl::PackedRGBComparison<PointT>::Ptr (new pcl::PackedRGBComparison<PointT> ("g", pcl::ComparisonOps::LE, gMax)));
	cond->addComparison(typename pcl::PackedRGBComparison<PointT>::Ptr (new pcl::PackedRGBComparison<PointT> ("b", pcl::ComparisonOps::GE, bMin)));
	cond->addComparison(typename pcl::PackedRGBComparison<PointT>::Ptr (new pcl::PackedRGBComparison<PointT> ("b", pcl::ComparisonOps::LE, bMax)));

	typename pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>);
	pcl::ConditionalRemoval<PointT> condrem;
	condrem.setCondition(cond);
	condrem.setInputCloud(cloud_in);
	condrem.setKeepOrganized(keep_org);
	condrem.filter(*cloud_out);

	return cloud_out;
}

template <typename PointT>
float minDistanceBetweenPointClouds(const typename pcl::PointCloud<PointT>::ConstPtr &cloud1, const typename pcl::PointCloud<PointT>::ConstPtr &cloud2, int &cp1, int &cp2) {
	cp1 = -1;
	cp2 = -1;
	float min_d = std::numeric_limits<float>::infinity();

	pcl::KdTreeFLANN<PointT> kdt;
	std::vector<int> neighbors(1);
	std::vector<float> distances(1);

	if (cloud1->size() > cloud2->size()) {
		kdt.setInputCloud(cloud1);
		for (size_t pointId = 0; pointId < cloud2->size(); pointId++) {
			kdt.nearestKSearch(cloud2->points[pointId], 1, neighbors, distances);
			if (distances[0] < min_d) {
				cp1 = neighbors[0];
				cp2 = pointId;
				min_d = distances[0];
			}
		}
	} else {
		kdt.setInputCloud(cloud2);
		for (size_t pointId = 0; pointId < cloud1->size(); pointId++) {
			kdt.nearestKSearch(cloud1->points[pointId], 1, neighbors, distances);
			if (distances[0] < min_d) {
				cp1 = pointId;
				cp2 = neighbors[0];
				min_d = distances[0];
			}
		}
	}

	return std::sqrt(min_d);
}

template<typename PointInT, typename PointOutT>
typename pcl::PointCloud<PointOutT>::Ptr estimatePointCloudNormals(const typename pcl::PointCloud<PointInT>::ConstPtr &cloud, float normal_radius) {
	typename pcl::PointCloud<PointOutT>::Ptr cloud_f (new pcl::PointCloud<PointOutT>);
	if (cloud->empty()) {
		return cloud_f;
	}
	pcl::PointCloud<pcl::Normal>::Ptr cloud_n (new pcl::PointCloud<pcl::Normal>);
	typename pcl::search::KdTree<PointInT>::Ptr tree(new pcl::search::KdTree<PointInT>());
	pcl::NormalEstimationOMP<PointInT, pcl::Normal> norm_est;
	norm_est.setSearchMethod(tree);
	norm_est.setRadiusSearch(normal_radius);
	norm_est.setInputCloud(cloud);
	norm_est.compute(*cloud_n);
	pcl::concatenateFields(*cloud, *cloud_n, *cloud_f);
	return cloud_f;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr downSamplePointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud_in, float res) {
	typename pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>);
	pcl::VoxelGrid<PointT> vg;
	vg.setLeafSize(res,res,res);
	vg.setInputCloud(cloud_in);
	vg.filter(*cloud_out);
	// std::vector<int> tmp;
	// pcl::removeNaNFromPointCloud(*cloud_out, *cloud_out, tmp);
	// pcl::removeNaNNormalsFromPointCloud(*cloud_out, *cloud_out, tmp);
	// for (int i = 0; i < cloud_out->size(); i++) {
	// 	std::cout << cloud_out->points[i].getNormalVector3fMap().norm() << std::endl;
	// 	cloud_out->points[i].getNormalVector3fMap().normalize();
	// }
	return cloud_out;
}

template<typename PointInT, typename PointOutT>
typename pcl::PointCloud<PointOutT>::Ptr smoothPointCloudMLS(const typename pcl::PointCloud<PointInT>::ConstPtr &cloud, float mls_radius, bool mls_poly_fit) {
	typename pcl::PointCloud<PointOutT>::Ptr cloud_mls (new pcl::PointCloud<PointOutT>);
	typename pcl::search::KdTree<PointInT>::Ptr tree(new pcl::search::KdTree<PointInT>());
	pcl::MovingLeastSquares<PointInT,PointOutT> mls;
	mls.setComputeNormals(true);
	mls.setInputCloud(cloud);
	mls.setPolynomialFit(mls_poly_fit);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(mls_radius);
	mls.process(*cloud_mls);

	std::vector<int> ind = mls.getCorrespondingIndices()->indices;
	for (int i = 0; i < ind.size(); i++) {
		Eigen::Vector3f normal_mls = cloud_mls->points[i].getNormalVector3fMap();
		Eigen::Vector3f normal = cloud->points[ind[i]].getNormalVector3fMap();
		if (normal_mls.dot(normal) < 0) {
			cloud_mls->points[i].getNormalVector3fMap() = -normal_mls;
		}
	}

	return cloud_mls;
}

template <typename PointT>
void projectPointCloudToPlane(typename pcl::PointCloud<PointT>::Ptr &cloud, const Eigen::Vector4f &plane) {
	Eigen::Vector3f normal = plane.head(3);
	float d = plane(3), nn = normal.norm();
	normal /= nn;
	d /= nn;
	for (size_t i = 0; i < cloud->size(); ++i) {
		Eigen::Vector3f point = cloud->points[i].getVector3fMap();
		cloud->points[i].getVector3fMap() = point - (normal.dot(point)+d)*normal;
	}
}

template <typename PointT>
void projectPointCloudToPlane(typename pcl::PointCloud<PointT>::Ptr &cloud, const std::vector<int> &indices, const Eigen::Vector4f &plane) {
	Eigen::Vector3f normal = plane.head(3);
	float d = plane(3), nn = normal.norm();
	normal /= nn;
	d /= nn;
	for (size_t i = 0; i < indices.size(); ++i) {
		Eigen::Vector3f point = cloud->points[indices[i]].getVector3fMap();
		cloud->points[indices[i]].getVector3fMap() = point - (normal.dot(point)+d)*normal;
	}
}

template<typename PointT>
void estimatePlaneFromPointCloudRANSAC(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, float dist_thresh, int iter, Eigen::Vector4f& plane, std::vector<int>& ind) {
	pcl::SACSegmentation<PointT> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	typename pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(iter);
	seg.setDistanceThreshold(dist_thresh);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

	plane = Eigen::Map<Eigen::Vector4f>(coefficients->values.data());
	ind = inliers->indices;
}

template<typename PointT>
int refineTransformICP(const typename pcl::PointCloud<PointT>::ConstPtr &src, const typename pcl::PointCloud<PointT>::ConstPtr &dst, Eigen::Matrix4f& tform, int iter, float dist_thresh) {

	Eigen::Matrix4f tform_init = tform;

	typename pcl::PointCloud<PointT>::Ptr src_t_final(new pcl::PointCloud<PointT>);

	boost::shared_ptr<pcl::registration::TransformationEstimationPointToPlaneLLS<PointT, PointT> > point_to_plane(new pcl::registration::TransformationEstimationPointToPlaneLLS<PointT, PointT>);

	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setTransformationEstimation(point_to_plane);
	icp.setMaximumIterations(iter);
	icp.setMaxCorrespondenceDistance(dist_thresh);
	icp.setInputSource(src);
	icp.setInputTarget(dst);
	icp.align(*src_t_final, tform_init);
	
	tform = icp.getFinalTransformation();
	tform = tform*tform_init;

	return icp.hasConverged();
}

template<typename PointT>
int refineTransformICP(const typename pcl::PointCloud<PointT>::ConstPtr &src, const typename pcl::PointCloud<PointT>::ConstPtr &dst, Eigen::Matrix3f& R, Eigen::Vector3f& t, int iter, float dist_thresh) {
	Eigen::Matrix4f tform = homogeneousTransformationMatrixFromRt(R,t);
	int success = refineTransformICP<PointT>(src, dst, tform, iter, dist_thresh);
	R = tform.block<3,3>(0,0);
	t = tform.block<3,1>(0,3);
	return success;
}

#endif /* POINTCLOUD_UTILITIES_HPP */
