#ifndef SISYPHUS_BOX_DETECTION_HPP
#define SISYPHUS_BOX_DETECTION_HPP

#include <chrono>
#include <random>

#include <sisyphus/graph_utilities.hpp>
#include <sisyphus/pointcloud_utilities.hpp>

#include <pcl/segmentation/extract_clusters.h>

#include <namaris/utilities/pcl_typedefs.hpp>
#include <namaris/utilities/map.hpp>
#include <namaris/utilities/pointcloud.hpp>
#include <namaris/algorithms/region_growing_normal_variation/region_growing_normal_variation.hpp>

template<typename PointT>
struct Box {
	Eigen::Matrix4f pose;
	Eigen::Vector3f size;
	std::vector<Eigen::Vector4f> planes;
	typename pcl::PointCloud<PointT>::Ptr pointCloud;
};

template<class T>
std::vector<T> extractVectorElementsFromIndices(const std::vector<T> &all, const std::vector<int> &subset) {
	std::vector<T> res(subset.size());
	for (int i = 0; i < subset.size(); i++) {
		res[i] = all[subset[i]];
	}
	return res;
}

template<typename PointT>
bool referenceVectorAngleTest(const PointT &p, const Eigen::Vector3f &ref_vector, const float angle_thresh) {
	return std::abs(p.getNormalVector3fMap().dot(ref_vector)) > std::cos(angle_thresh);
}

template <typename PointT>
void extractPlanarSegmentsRANSAC(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, std::vector<typename pcl::PointCloud<PointT>::Ptr> &segments, std::vector<Eigen::Vector4f> &planes, int cluster_size, float dist_thresh, float radius, float normal_var_thresh, float angle_thresh, int iter) {
	segments.clear();
	planes.clear();

	typename pcl::PointCloud<PointT>::Ptr pc(new pcl::PointCloud<PointT>);
	*pc = *cloud;

	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(iter);
	seg.setDistanceThreshold(dist_thresh);
	seg.setInputCloud(pc);

	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(pc);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr plane_coeff(new pcl::ModelCoefficients ());

	std::vector<Eigen::Vector4f> planes_tmp;

	while (pc->size() > cluster_size) {
		seg.segment(*inliers, *plane_coeff);

		if (inliers->indices.size() > cluster_size) {
			extract.setIndices(inliers);
			extract.setNegative(true);
			extract.filter(*pc);

			Eigen::Vector4f plane = Eigen::Vector4f(plane_coeff->values.data());
			float norm = plane.head(3).norm();
			plane = plane/norm;

			planes_tmp.push_back(plane);
		} else {
			break;
		}
	}

	for (int i = 0; i < planes_tmp.size(); i++) {
		std::vector<Eigen::Vector4f> constr;
		constr.push_back(planes_tmp[i] + Eigen::Vector4f(0,0,0,dist_thresh));
		constr.push_back(-planes_tmp[i] + Eigen::Vector4f(0,0,0,dist_thresh));

		std::vector<int> pt_ind;
		pointIndicesInConvexPolytope<PointT>(cloud, constr, pt_ind);
		typename pcl::PointCloud<PointT>::Ptr planar_pc = extractPointCloudFromIndices<PointT>(cloud, pt_ind, false, false);

		std::vector<int> tmp;
		pcl::removeNaNFromPointCloud(*planar_pc, *planar_pc, tmp);
		pcl::removeNaNNormalsFromPointCloud(*planar_pc, *planar_pc, tmp);

		utl::map::Map clusters;
		// float angle_thresh = 10.0*M_PI/180.0;
		// float normal_var_thresh = 100*30.0*M_PI/180.0;

		boost::function<bool (const PointT&)> unary_func = boost::bind(referenceVectorAngleTest<PointT>, _1, planes_tmp[i].head(3), angle_thresh);

		alg::RegionGrowingNormalVariation<PointT> rg;
		rg.setInputCloud(planar_pc);
		rg.setConsistentNormals(true);
		rg.setNumberOfNeighbors(10);
		rg.setSearchRadius(radius);
		rg.setNormalVariationThreshold(normal_var_thresh);
		rg.setMinValidBinaryNeighborsFraction(0.5);
		rg.setUnaryConditionFunction(unary_func);
		rg.setMinSegmentSize(cluster_size);
		rg.segment(clusters);

		for (int j = 0; j < clusters.size(); j++) {
			typename pcl::PointCloud<PointT>::Ptr pc_tmp = extractPointCloudFromIndices<PointT>(planar_pc, clusters[j], false, false);

			// Eigen::Vector4f plane = planes_tmp[i];
			seg.setInputCloud(pc_tmp);
			seg.segment(*inliers, *plane_coeff);
			Eigen::Vector4f plane = Eigen::Vector4f(plane_coeff->values.data());

			Eigen::Vector3f plane_normal = plane.head(3);
			int pos = 0, neg = 0;
			for (int k = 0; k < pc_tmp->size(); k++) {
				if (pc_tmp->points[k].getNormalVector3fMap().dot(plane_normal) < 0) {
					neg++;
				} else {
					pos++;
				}
			}
			if (neg > pos) {
				plane = -plane;
			}

			segments.push_back(pc_tmp);
			planes.push_back(plane);
		}
	}

}

template <typename PointT>
void extractPlanarSegmentsConstrained(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<Eigen::Vector3f> &modes, std::vector<typename pcl::PointCloud<PointT>::Ptr> &segments, std::vector<Eigen::Vector4f> &planes, int cluster_size, float dist_thresh, float radius, float normal_var_thresh, float angle_thresh, int iter) {
	segments.clear();
	planes.clear();

	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(iter);
	seg.setDistanceThreshold(dist_thresh);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr plane_coeff(new pcl::ModelCoefficients ());

	for (int i = 0; i < modes.size(); i++) {
		utl::map::Map clusters;
		// float angle_thresh = 10.0*M_PI/180.0;
		// float normal_var_thresh = 100*30.0*M_PI/180.0;

		boost::function<bool (const PointT&)> unary_func = boost::bind(referenceVectorAngleTest<PointT>, _1, modes[i], angle_thresh);

		alg::RegionGrowingNormalVariation<PointT> rg;
		rg.setInputCloud(cloud);
		rg.setConsistentNormals(true);
		rg.setNumberOfNeighbors(10);
		rg.setSearchRadius(radius);
		rg.setNormalVariationThreshold(normal_var_thresh);
		rg.setMinValidBinaryNeighborsFraction(0.5);
		rg.setUnaryConditionFunction(unary_func);
		rg.setMinSegmentSize(cluster_size);
		rg.segment(clusters);

		for (int j = 0; j < clusters.size(); j++) {
			typename pcl::PointCloud<PointT>::Ptr pc_tmp = extractPointCloudFromIndices<PointT>(cloud, clusters[j], false, false);

			seg.setInputCloud(pc_tmp);
			seg.segment(*inliers, *plane_coeff);
			Eigen::Vector4f plane = Eigen::Vector4f(plane_coeff->values.data());

			Eigen::Vector3f plane_normal = plane.head(3);
			int pos = 0, neg = 0;
			for (int k = 0; k < pc_tmp->size(); k++) {
				if (pc_tmp->points[k].getNormalVector3fMap().dot(plane_normal) < 0) {
					neg++;
				} else {
					pos++;
				}
			}
			if (neg > pos) {
				plane = -plane;
			}

			segments.push_back(pc_tmp);
			planes.push_back(plane);
		}
	}
}

template <typename PointT>
void refinePlanarSegments(std::vector<typename pcl::PointCloud<PointT>::Ptr> &segments, std::vector<Eigen::Vector4f> &planes, int cluster_size, float dist_thresh, float radius, float normal_var_thresh, float angle_thresh, int iter) {

	std::vector<typename pcl::PointCloud<PointT>::Ptr> segments_new, segments_curr;
	std::vector<Eigen::Vector4f> planes_new, planes_curr;

	for (int i = 0; i < segments.size(); ++i) {
		extractPlanarSegmentsRANSAC<PointT>(segments[i], segments_curr, planes_curr, cluster_size, dist_thresh, radius, normal_var_thresh, angle_thresh, iter);
		// if (segments_curr.size() == 0) {
		// 	segments_new.push_back(segments[i]);
		// 	planes_new.push_back(planes[i]);
		// 	continue;
		// }
		// if (segments_curr.size() == 1) {
		// 	segments_new.push_back(segments[i]);
		// 	planes_new.push_back(planes[i]);
		// 	continue;
		// }
		for (int j = 0; j < segments_curr.size(); ++j) {
			segments_new.push_back(segments_curr[j]);
			planes_new.push_back(planes_curr[j]);
		}
	}

	segments = segments_new;
	planes = planes_new;
}

template<typename PointT>
pcl::PointCloud<pcl::PointXYZ>::Ptr getNormalCloudAsPointXYZ(const typename pcl::PointCloud<PointT>::ConstPtr &cloud) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_n(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud->size(); ++i) {
		pcl::PointXYZ pt;
		pt.getVector3fMap() = cloud->points[i].getNormalVector3fMap();
		cloud_n->push_back(pt);
	}
	return cloud_n;
}

template<typename PointT>
void findCoordinateModes(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<float> &pt_w, std::vector<Eigen::Vector3f> &modes, std::vector<float> &modes_w, float ang_thresh) {
	modes.clear();
	modes_w.clear();

	if (cloud->empty()) {
		return;
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::KdTreeFLANN<PointT> kdt;
	kdt.setInputCloud(cloud);

	pcl::PointXYZ curr_pt, prev_pt;
	std::vector<int> pt_ind;
	std::vector<float> pt_sq_dist;
	int num_pts, curr_count;
	float upd_dist;
	for (int i = 0; i < cloud->size(); ++i) {
		curr_pt = cloud->points[i];

		do {
			prev_pt = curr_pt;

			num_pts = kdt.radiusSearch(curr_pt, ang_thresh, pt_ind, pt_sq_dist);

			Eigen::MatrixXf wnbs(3, num_pts);
			curr_count = 0;
			for (int j = 0; j < pt_ind.size(); ++j) {
				wnbs.col(j) = pt_w[pt_ind[j]]*cloud->points[pt_ind[j]].getVector3fMap();
				curr_count += pt_w[pt_ind[j]];
			}
			curr_pt.getVector3fMap() = wnbs.rowwise().sum().normalized();

			upd_dist = (curr_pt.getVector3fMap() - prev_pt.getVector3fMap()).norm();
		} while (upd_dist >= 0.000001);

		cloud_tmp->push_back(curr_pt);
	}

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_tmp);
	std::vector<pcl::PointIndices> clusters;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(ang_thresh);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_tmp);
	ec.extract(clusters);

	for (int i = 0; i < clusters.size(); ++i) {
		Eigen::MatrixXf wnbs(3, clusters[i].indices.size());
		curr_count = 0;
		for (int j = 0; j < clusters[i].indices.size(); ++j) {
			wnbs.col(j) = pt_w[clusters[i].indices[j]]*cloud_tmp->points[clusters[i].indices[j]].getVector3fMap();
			curr_count += pt_w[clusters[i].indices[j]];
		}
		modes.push_back(wnbs.rowwise().sum().normalized());
		modes_w.push_back(curr_count);
		// std::cout << std::endl << (float)curr_count/clusters[i].indices.size() << std::endl;
		// std::cout << "   " << curr_count << std::endl;
	}
}


template<typename PointT>
void findHighDensityAreas(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out, std::vector<float> &pt_w) {

	cloud_out = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pt_w.clear();

	if (cloud->empty()) {
		return;
	}

	int n_samples = 10000;
	float ang_thresh = 2.0/std::sqrt((float)n_samples);
	int min_knn = 3.0*(float)cloud->size()*ang_thresh*ang_thresh/4.0;

	// std::cout << n_samples << std::endl;
	// std::cout << ang_thresh << std::endl;
	// std::cout << min_knn << std::endl;

	pcl::KdTreeFLANN<PointT> kdt;
	kdt.setInputCloud(cloud);

	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::normal_distribution<float> randn(0.0,1.0);

	pcl::PointXYZ curr_pt;
	std::vector<int> pt_ind;
	std::vector<float> pt_sq_dist;
	int num_pts;
	for (int i = 0; i < n_samples; ++i) {
		do {
			curr_pt.x = randn(rng);
			curr_pt.y = randn(rng);
			curr_pt.z = randn(rng);
		} while (curr_pt.x == 0.0 && curr_pt.y == 0.0 && curr_pt.z == 0.0);
		curr_pt.getVector3fMap().normalize();

		num_pts = kdt.radiusSearch(curr_pt, ang_thresh, pt_ind, pt_sq_dist);
		if (num_pts >= min_knn) {
			cloud_out->push_back(curr_pt);
			pt_w.push_back(num_pts);
		}
	}
}

template<typename PointT>
void normalModesFromPointCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, std::vector<Eigen::Vector3f> &modes, std::vector<float> &modes_w, float ang_thresh) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_n = getNormalCloudAsPointXYZ<PointT>(cloud);

	// viewer->removeAllPointClouds();
	// viewer->addPointCloud(cloud_n, "normals");
	// viewer->spin();

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_d;
	std::vector<float> pt_w_d;
	findHighDensityAreas<pcl::PointXYZ>(cloud_n, cloud_d, pt_w_d);

	// std::cout << pt_w_d.size() << " points" << std::endl;
	// viewer->removeAllPointClouds();
	// viewer->addPointCloud(cloud_d, "normals");
	// viewer->spin();

	findCoordinateModes<pcl::PointXYZ>(cloud_d, pt_w_d, modes, modes_w, ang_thresh);

	// std::cout << modes_w.size() << " modes" << std::endl;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
	// for (int i = 0; i < modes_w.size(); ++i) {
	// 	pcl::PointXYZ pt;
	// 	pt.getVector3fMap() = modes[i];
	// 	tmp->push_back(pt);
	// 	// std::cout << "W: " << modes_w[i] << std::endl;
	// }
	// viewer->removeAllPointClouds();
	// viewer->addPointCloud(tmp, "modes");
	// viewer->spin();
}

std::vector<Eigen::Vector3f> extractNormalsWithOrthogonalCounterpart(const std::vector<Eigen::Vector3f> &normals, float angle_thresh);

template<typename PointT>
char pairwisePlanarSegmentMergeCheck(const typename pcl::PointCloud<PointT>::ConstPtr &seg1, const Eigen::Vector4f &plane1, const typename pcl::PointCloud<PointT>::ConstPtr &seg2, const Eigen::Vector4f &plane2, float angle_thresh, float dist_thresh, float merge_tol_dist) {

	Eigen::Vector3f n1 = plane1.head(3), n2 = plane2.head(3);
	if (n1.dot(n2) < std::cos(angle_thresh)) {
		return 0;
	}

	Eigen::MatrixXf pts1 = seg1->getMatrixXfMap().topRows(3);
	Eigen::MatrixXf pts2 = seg2->getMatrixXfMap().topRows(3);

	Eigen::MatrixXf dist1 = ((n2.transpose()*pts1).array() + plane2(3)).array().abs();
	Eigen::MatrixXf dist2 = ((n1.transpose()*pts2).array() + plane1(3)).array().abs();

	if (dist1.mean() > dist_thresh && dist2.mean() > dist_thresh) {
		return 0;
	}

	int p1, p2;
	if (minDistanceBetweenPointClouds<PointT>(seg1, seg2, p1, p2) > merge_tol_dist) {
		return 0;
	}

	return 1;
}

template<typename PointT>
void mergePlanarSegments(std::vector<typename pcl::PointCloud<PointT>::Ptr> &segments, std::vector<Eigen::Vector4f> &planes, float angle_thresh, float dist_thresh, float merge_tol_dist, int iter) {

	int n_segments = planes.size();

	std::vector<std::vector<char> > adj_mat(n_segments);
	for (int i = 0; i < adj_mat.size(); ++i) {
		adj_mat[i] = std::vector<char>(n_segments);
	}
	for (int i = 0; i < adj_mat.size(); ++i) {
		for (int j = i+1; j < adj_mat[i].size(); ++j) {
			adj_mat[i][j] = pairwisePlanarSegmentMergeCheck<PointT>(segments[i], planes[i], segments[j], planes[j], angle_thresh, dist_thresh, merge_tol_dist);
			adj_mat[j][i] = adj_mat[i][j];
		}
	}

	std::vector<std::vector<int> > clusters = findConnectedComponentsDFS(adj_mat);

	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(iter);
	seg.setDistanceThreshold(dist_thresh);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr plane_coeff(new pcl::ModelCoefficients ());

	std::vector<typename pcl::PointCloud<PointT>::Ptr> segments_new(clusters.size());
	std::vector<Eigen::Vector4f> planes_new(clusters.size());

	typename pcl::PointCloud<PointT>::Ptr curr_seg;
	for (int i = 0; i < clusters.size(); ++i) {
		if (clusters[i].size() == 1) {
			segments_new[i] = segments[clusters[i][0]];
			planes_new[i] = planes[clusters[i][0]];
		} else {
			curr_seg = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
			for (int j = 0; j < clusters[i].size(); ++j) {
				*curr_seg += *segments[clusters[i][j]];
			}
			
			seg.setInputCloud(curr_seg);
			seg.segment(*inliers, *plane_coeff);

			Eigen::Vector4f curr_plane = Eigen::Vector4f(plane_coeff->values.data());
			Eigen::Vector3f n_curr = curr_plane.head(3);
			curr_plane /= n_curr.norm();
			if (curr_plane.dot(planes[clusters[i][0]]) < 0.0) {
				curr_plane = -curr_plane;
			}
			
			segments_new[i] = curr_seg;
			planes_new[i] = curr_plane;
		}
	}
	segments = segments_new;
	planes = planes_new;
}

template<typename PointT>
char pairwiseBoxSideCheck(const typename pcl::PointCloud<PointT>::ConstPtr &seg1, const Eigen::Vector4f &plane1, const typename pcl::PointCloud<PointT>::ConstPtr &seg2, const Eigen::Vector4f &plane2, float angle_thresh, float dist_thresh, float ratio_thresh, float range_thresh, bool outside) {

	Eigen::Vector4f p1 = plane1, p2 = plane2;
	if (!outside) {
		p1 = -p1;
		p2 = -p2;
	}

	bool ortho_test = std::abs(p1.head(3).dot(p2.head(3))) < std::sin(angle_thresh);

	if (!ortho_test) {
		bool par_test = std::abs(p1.head(3).dot(p2.head(3))) > std::cos(angle_thresh);
		if (par_test) {
			return 2;
		}
		return 0;
	}

	// std::vector<Eigen::Vector4f> constr1, constr2;
	// constr1.push_back(p1);
	// constr2.push_back(p2);

	std::vector<Eigen::Vector4f> constr1(1, p1), constr2(1, p2);
	std::vector<int> ind1, ind2;
	pointIndicesInConvexPolytope<PointT>(seg2, constr1, ind1);
	pointIndicesInConvexPolytope<PointT>(seg1, constr2, ind2);

	bool orient_test = (ind1.size() < ratio_thresh*(float)seg2->size()) && (ind2.size() < ratio_thresh*(float)seg1->size());

	if (!orient_test) {
		return 0;
	}

	Eigen::MatrixXf pts1 = seg1->getMatrixXfMap().topRows(3);
	Eigen::MatrixXf pts2 = seg2->getMatrixXfMap().topRows(3);

	Eigen::Vector3f vec = p1.block<3,1>(0,0).cross(p2.block<3,1>(0,0)).normalized();
	Eigen::MatrixXf proj1 = vec.transpose()*pts1;
	Eigen::MatrixXf proj2 = vec.transpose()*pts2;

	float v1min = proj1.minCoeff(), v1max = proj1.maxCoeff(), v2min = proj2.minCoeff(), v2max = proj2.maxCoeff();

	float overlap = 0.0, span = std::max(v1max,v2max) - std::min(v1min,v2min);
	if (v1max >= v2min && v2max >= v1min) {
		overlap = std::min(v1max,v2max) - std::max(v1min,v2min);
	}

	bool range_test = overlap/span > range_thresh;

	if (!range_test) {
		return 0;
	}

	int i1, i2;
	bool dist_test = minDistanceBetweenPointClouds<PointT>(seg1,seg2,i1,i2) < dist_thresh;

	if (!dist_test) {
		return 0;
	}

	return 1;
}

template<typename PointT>
std::vector<std::vector<int> > extractBoxSegmentClusters(const std::vector<typename pcl::PointCloud<PointT>::Ptr> &segments, const std::vector<Eigen::Vector4f> &planes, float angle_thresh, float dist_thresh, float ratio_thresh, float range_thresh, bool outside, bool require_cliques = false) {

	int n_segments = planes.size();

	std::vector<std::vector<char> > adj_mat(n_segments);
	for (int i = 0; i < adj_mat.size(); ++i) {
		adj_mat[i] = std::vector<char>(n_segments);
	}
	for (int i = 0; i < adj_mat.size(); ++i) {
		for (int j = i+1; j < adj_mat[i].size(); ++j) {
			adj_mat[i][j] = pairwiseBoxSideCheck<PointT>(segments[i], planes[i], segments[j], planes[j], angle_thresh, dist_thresh, ratio_thresh, range_thresh, outside);
			adj_mat[j][i] = adj_mat[i][j];
		}
	}

	// for (int i = 0; i < adj_mat.size(); ++i) {
	// 	for (int j = 0; j < adj_mat[i].size(); ++j) {
	// 		std::cout << (int)adj_mat[i][j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	std::vector<std::vector<int> > clusters, box_clusters;
	if (require_cliques) {
		clusters = findCliques(adj_mat);
	} else {
		clusters = findConnectedComponentsDFS(adj_mat);
	}
	for (int i = 0; i < clusters.size(); ++i) {
		if (clusters[i].size() > 1) {
			box_clusters.push_back(clusters[i]);
		}
	}

	return box_clusters;
}

Eigen::Matrix3f getBoxAxesFromSideNormals(const std::vector<Eigen::Vector3f> &side_normals, float angle_thresh);

template<typename PointT>
Box<PointT> extractBoxFromSegmentCluster(const std::vector<typename pcl::PointCloud<PointT>::Ptr> &segments, const std::vector<Eigen::Vector4f> &planes, float angle_thresh, bool outside) {
	
	Box<PointT> box;

	std::vector<Eigen::Vector4f> planes_o = planes;
	if (!outside) {
		for (int i = 0; i < planes_o.size(); ++i) {
			planes_o[i] = -planes_o[i];
		}
	}

	std::vector<Eigen::Vector3f> segment_normals;
	for (int i = 0; i < planes.size(); ++i) {
		segment_normals.push_back(planes_o[i].head(3).normalized());
	}
	Eigen::Matrix3f R = getBoxAxesFromSideNormals(segment_normals, angle_thresh);

	typename pcl::PointCloud<PointT>::Ptr box_pcd(new pcl::PointCloud<PointT>);
	for (int i = 0; i < segments.size(); ++i) {
		*box_pcd += *segments[i];
	}

	Eigen::MatrixXf pts = box_pcd->getMatrixXfMap().topRows(3);

	pts = R.transpose()*pts;
	Eigen::Vector3f pt_min = pts.rowwise().minCoeff(), pt_max = pts.rowwise().maxCoeff();

	std::vector<Eigen::Vector4f> side_planes(6);
	for (int i = 0; i < 6; ++i) {
		Eigen::Vector3f normal = R.col(i/2);
		float offset;
		if (i%2 == 1) {
			normal = -normal;
		}
		std::vector<typename pcl::PointCloud<PointT>::Ptr> curr_side_clouds;
		for (int j = 0; j < segment_normals.size(); ++j) {
			if (normal.dot(segment_normals[j]) > std::cos(angle_thresh)) {
				curr_side_clouds.push_back(segments[j]);
			}
		}
		if (curr_side_clouds.empty()) {
			if (i%2 == 1) {
				offset = pt_min(i/2);
			} else {
				offset = pt_max(i/2);
			}
		} else {
			typename pcl::PointCloud<PointT>::Ptr curr_side_cloud(new pcl::PointCloud<PointT>);
			for (int j = 0; j < curr_side_clouds.size(); ++j) {
				*curr_side_cloud += *curr_side_clouds[j];
			}

			Eigen::MatrixXf pts_curr = curr_side_cloud->getMatrixXfMap().topRows(3);

			pts_curr = R.transpose()*pts_curr;
			offset = pts_curr.row(i/2).mean();
			if (i%2 == 1) {
				pt_min(i/2) = offset;
			} else {
				pt_max(i/2) = offset;
			}
		}
		side_planes[i].head(3) = R.col(i/2);
		side_planes[i](3) = -offset;
		if (i%2 == 1) {
			side_planes[i] = -side_planes[i];
		}
	}

	if (!outside) {
		for (int i = 0; i < side_planes.size(); ++i) {
			side_planes[i] = -side_planes[i];
		}
	}

	Eigen::Vector3f t = R*(pt_max+pt_min)/2.0;
	box.size = pt_max-pt_min;
	box.pose = homogeneousTransformationMatrixFromRt(R,t);
	box.pointCloud = box_pcd;
	box.planes = side_planes;

	return box;
}

template<typename PointT>
void renameBoxAxes(Box<PointT> &box, const Eigen::Vector3f &origin, const Eigen::Vector3f &z_ref) {
	Eigen::Vector3f x_ref = (origin - box.pose.block(0,3,3,1)).normalized();
	Eigen::Matrix3f axes_old = box.pose.block(0,0,3,3), axes_new;
	std::vector<int> axes_perm(3), axes_sign(3);

	float max_val = -1;
	for (int i = 0; i < 3; ++i) {
		float dp = z_ref.dot(axes_old.col(i));
		float val = std::abs(dp);
		if (val > max_val) {
			axes_perm[2] = i;
			axes_sign[2] = (dp < 0) ? -1 : 1;
			max_val = val;
		}
	}

	max_val = -1;
	for (int i = 0; i < 3; ++i) {
		float dp = x_ref.dot(axes_old.col(i));
		float val = std::abs(dp);
		if (val > max_val && i != axes_perm[2]) {
			axes_perm[0] = i;
			axes_sign[0] = (dp < 0) ? -1 : 1;
			max_val = val;
		}
	}

	Eigen::Vector3f y_ref = (axes_sign[2]*axes_old.col(axes_perm[2])).cross(axes_sign[0]*axes_old.col(axes_perm[0]));

	for (int i = 0; i < 3; ++i) {
		if (i != axes_perm[0] && i != axes_perm[2]) {
			float dp = y_ref.dot(axes_old.col(i));
			axes_perm[1] = i;
			axes_sign[1] = (dp < 0) ? -1 : 1;
		}
	}

	Eigen::Vector3f sz = box.size;
	for (int i = 0; i < 3; ++i) {
		axes_new.col(i) = axes_sign[i]*axes_old.col(axes_perm[i]);
		box.size(i) = sz(axes_perm[i]);
	}
	box.pose.block(0,0,3,3) = axes_new;

	std::vector<Eigen::Vector4f> planes = box.planes;
	for (int i = 0; i < 6; ++i) {
		int ind = 2*axes_perm[i/2] + ((axes_sign[i/2] > 0) ? i%2 : 1-i%2);
		box.planes[i] = planes[ind];
	}

	// std::cout << "PERMUTATION: " << axes_perm[0] << " " << axes_perm[1] << " " << axes_perm[2] << std::endl;
	// std::cout << "SIGN: " << axes_sign[0] << " " << axes_sign[1] << " " << axes_sign[2] << std::endl;

	// std::cout << "BEFORE:" << std::endl << axes_old << std::endl;
	// std::cout << "   DET: " << axes_old.determinant() << std::endl;
	// std::cout << "AFTER:" << std::endl << axes_new << std::endl;
	// std::cout << "   DET: " << axes_new.determinant() << std::endl;

	// std::cout << "BEFORE:" << std::endl;
	// for (int i = 0; i < 6; ++i) {
	// 	std::cout << planes[i].transpose() << std::endl;
	// }
	// std::cout << "AFTER:" << std::endl;
	// for (int i = 0; i < 6; ++i) {
	// 	std::cout << box.planes[i].transpose() << std::endl;
	// }
}

template<typename PointT>
std::vector<Box<PointT> > detectBoxes(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, int min_cluster_size, int ransac_iter, float max_plane_dist, float max_seg_dist, float rg_radius, float rg_normal_var_thresh, float rg_angle_thresh, float angle_tol, float max_angle_diff, float angle_diff_cluster, float max_side_dist, float ratio_thresh, float range_thresh, bool outside, bool require_cliques = false, bool brute_force = false) {

	// // Segmenting
	// int min_cluster_size = 500;
	// int ransac_iter = 1000;
	// float max_plane_dist = 0.015;
	// float max_seg_dist = 0.2;
	// float rg_radius = 0.02;
	// float rg_normal_var_thresh = 100*30.0*M_PI/180.0;
	// float rg_angle_thresh = 30.0*M_PI/180.0;
	// float angle_tol = 5.0*M_PI/180.0;
	// float max_angle_diff = 15.0*M_PI/180.0;
	// float angle_diff_cluster = 10.0*M_PI/180.0;

	// // Grouping
	// float max_side_dist = 0.1;
	// float ratio_thresh = 0.05;
	// float range_thresh = 0.65;
	// bool outside = true;
	// bool require_cliques = false;
	// bool brute_force = false;

	std::vector<Eigen::Vector3f> modes;
	std::vector<float> modes_w;
	std::vector<typename pcl::PointCloud<PointT>::Ptr> segments;
	std::vector<Eigen::Vector4f> planes;

	if (brute_force) {
		// Brute force RANSAC plane fitting
		extractPlanarSegmentsRANSAC<PointT>(cloud, segments, planes, min_cluster_size, max_plane_dist, rg_radius, rg_normal_var_thresh, rg_angle_thresh, ransac_iter);
	} else {
		normalModesFromPointCloud<PointT>(cloud, modes, modes_w, angle_diff_cluster);
		modes = extractNormalsWithOrthogonalCounterpart(modes, angle_tol);
		extractPlanarSegmentsConstrained<PointT>(cloud, modes, segments, planes, min_cluster_size, max_plane_dist, rg_radius, rg_normal_var_thresh, rg_angle_thresh, ransac_iter);
	}

	refinePlanarSegments<PointT>(segments, planes, min_cluster_size, max_plane_dist, rg_radius, rg_normal_var_thresh, max_angle_diff, ransac_iter);
	mergePlanarSegments<PointT>(segments, planes, angle_tol, max_plane_dist, max_seg_dist, ransac_iter);

	// int n_seg;
	// do {
	// 	n_seg = segments.size();
	// 	mergePlanarSegments<PointT>(segments, planes, angle_tol, max_plane_dist, max_seg_dist, ransac_iter);
	// 	std::cout << "MERGE" << std::endl;
	// } while (n_seg != segments.size());

	std::vector<std::vector<int> > box_clusters = extractBoxSegmentClusters<PointT>(segments, planes, angle_tol, max_side_dist, ratio_thresh, range_thresh, outside, require_cliques);

	std::vector<Box<PointT> > boxes(box_clusters.size());
	for (int i = 0; i < box_clusters.size(); ++i) {
		std::vector<typename pcl::PointCloud<PointT>::Ptr> seg_curr = extractVectorElementsFromIndices<typename pcl::PointCloud<PointT>::Ptr>(segments, box_clusters[i]);
		std::vector<Eigen::Vector4f> planes_curr = extractVectorElementsFromIndices<Eigen::Vector4f>(planes, box_clusters[i]);
		boxes[i] = extractBoxFromSegmentCluster<PointT>(seg_curr, planes_curr, angle_tol, outside);
		// renameBoxAxes<PointT>(boxes[i], Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitZ());
	}

	return boxes;
}

#endif /* SISYPHUS_BOX_DETECTION_HPP */
