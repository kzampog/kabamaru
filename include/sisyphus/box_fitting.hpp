#ifndef SISYPHUS_BOX_FITTING_HPP
#define SISYPHUS_BOX_FITTING_HPP

#include <set>
#include <sisyphus/pointcloud_utilities.hpp>
#include <pcl/surface/convex_hull.h>

template<typename PointT>
bool fitMinimumVolumeBoundingBoxConstrained(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const Eigen::Vector4f &support_plane, Eigen::Affine3f &pose, Eigen::Vector3f &size) {

	if (cloud->empty()) return false;

	typename pcl::PointCloud<PointT>::Ptr cloud_proj(new pcl::PointCloud<PointT>);
	*cloud_proj = *cloud;
	projectPointCloudToPlane<PointT>(cloud_proj, support_plane);

	typename pcl::PointCloud<PointT>::Ptr hull(new pcl::PointCloud<PointT>);
	pcl::ConvexHull<PointT> ch;
	ch.setDimension(2);
	ch.setInputCloud(cloud_proj);
	ch.reconstruct(*hull);

	Eigen::MatrixXf hull_pts = hull->getMatrixXfMap().topRows(3);
	Eigen::MatrixXf obj_pts = cloud->getMatrixXfMap().topRows(3);

	Eigen::Vector3f z_dir = support_plane.head(3).normalized();

	Eigen::Vector3f min_pt, max_pt;
	Eigen::Matrix3f R, R_tmp;
	R_tmp.row(2) = z_dir;

	float vol_tmp, min_vol = std::numeric_limits<float>::infinity();
	for (int i0 = 0; i0 < hull_pts.cols(); ++i0) {
		int i1 = (i0+1)%hull_pts.cols();
		R_tmp.row(0) = (hull_pts.col(i1)-hull_pts.col(i0)).normalized();
		R_tmp.row(1) = (z_dir.cross(R_tmp.row(0))).normalized();

		Eigen::MatrixXf obj_pts_tmp = R_tmp*obj_pts;

		Eigen::Vector3f min_pt_tmp = obj_pts_tmp.rowwise().minCoeff();
		min_pt_tmp(2) = -support_plane(3);
		Eigen::Vector3f max_pt_tmp = obj_pts_tmp.rowwise().maxCoeff();
		vol_tmp = (max_pt_tmp-min_pt_tmp).prod();

		if (vol_tmp < min_vol) {
			min_vol = vol_tmp;
			R = R_tmp.transpose();
			min_pt = min_pt_tmp;
			max_pt = max_pt_tmp;
		}
	}

	Eigen::Vector3f t = R*(max_pt+min_pt)/2.0;

	pose.linear() = R;
	pose.translation() = t;
	size = max_pt-min_pt;

	return true;
}

template<typename PointT>
bool fitMaximumContactBoundingBoxConstrainedIterative(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const Eigen::Vector4f &support_plane, Eigen::Affine3f &pose, Eigen::Vector3f &size, int &num_inliers, float &mean_inlier_dist, const Eigen::Vector3f &initial_dir, float inlier_dist, int max_iter = 200, float tol = 1e-5) {

	typename pcl::PointCloud<PointT>::Ptr cloud_3d(new pcl::PointCloud<PointT>);
	*cloud_3d = *cloud;

	// Check if cloud is empty
	if (cloud_3d->empty()) return false;

	Eigen::Vector4f sup_plane = support_plane/support_plane.head(3).norm();

	// Project cloud points to support plane
	typename pcl::PointCloud<PointT>::Ptr cloud_2d(new pcl::PointCloud<PointT>);
	*cloud_2d = *cloud_3d;
	projectPointCloudToPlane<PointT>(cloud_2d, sup_plane);
	Eigen::MatrixXf pts_3d = cloud_3d->getMatrixXfMap().topRows(3);
	Eigen::MatrixXf pts_2d = cloud_2d->getMatrixXfMap().topRows(3);

	// Compute initial pose
	Eigen::Vector3f z_dir = sup_plane.head(3);
	Eigen::Vector3f x_dir = initial_dir.normalized();
	x_dir = (x_dir - x_dir.dot(z_dir)*z_dir).normalized();
	Eigen::Vector3f y_dir = (z_dir.cross(x_dir)).normalized();

	Eigen::Matrix3f R, R_prev;
	R.row(0) = x_dir;
	R.row(1) = y_dir;
	R.row(2) = z_dir;

	std::set<std::vector<float> > search_traj;
	std::vector<float> Rv(9);
	Eigen::Map<Eigen::Matrix3f>(Rv.data()) = R;
	search_traj.insert(Rv);

	int iter = 0;
	while (true) {
		R_prev = R;

		Eigen::MatrixXf pts_3d_trans = R*pts_3d;
		Eigen::Vector3f min_pt_trans = pts_3d_trans.rowwise().minCoeff();
		min_pt_trans(2) = -sup_plane(3);
		Eigen::Vector3f max_pt_trans = pts_3d_trans.rowwise().maxCoeff();

		// Find side inliers
		Eigen::MatrixXf xmin_dist = pts_3d_trans.row(0).array() - min_pt_trans(0);
		Eigen::MatrixXf xmax_dist = max_pt_trans(0) - pts_3d_trans.row(0).array();
		Eigen::MatrixXf ymin_dist = pts_3d_trans.row(1).array() - min_pt_trans(1);
		Eigen::MatrixXf ymax_dist = max_pt_trans(1) - pts_3d_trans.row(1).array();

		Eigen::MatrixXf xmin_inliers(3, (xmin_dist.array() <= inlier_dist).count());
		Eigen::MatrixXf xmax_inliers(3, (xmax_dist.array() <= inlier_dist).count());
		Eigen::MatrixXf ymin_inliers(3, (ymin_dist.array() <= inlier_dist).count());
		Eigen::MatrixXf ymax_inliers(3, (ymax_dist.array() <= inlier_dist).count());

		int xmin_ind = 0, xmax_ind = 0, ymin_ind = 0, ymax_ind = 0;
		for (int i = 0; i < pts_2d.cols(); ++i) {
			if (xmin_dist(i) <= inlier_dist) xmin_inliers.col(xmin_ind++) = pts_2d.col(i);
			if (xmax_dist(i) <= inlier_dist) xmax_inliers.col(xmax_ind++) = pts_2d.col(i);
			if (ymin_dist(i) <= inlier_dist) ymin_inliers.col(ymin_ind++) = pts_2d.col(i);
			if (ymax_dist(i) <= inlier_dist) ymax_inliers.col(ymax_ind++) = pts_2d.col(i);
		}

		std::vector<Eigen::Vector3f> x_dirs(0), y_dirs(0);
		std::vector<int> x_dir_w(0), y_dir_w(0);

		if (xmin_inliers.cols() > 2) {
			Eigen::Vector3f mu = xmin_inliers.rowwise().mean();
			xmin_inliers = xmin_inliers.colwise() - mu;
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(xmin_inliers, Eigen::ComputeFullU | Eigen::ComputeFullV);
			y_dirs.push_back(svd.matrixU().col(0));
			y_dir_w.push_back(xmin_inliers.cols());
		}

		if (xmax_inliers.cols() > 2) {
			Eigen::Vector3f mu = xmax_inliers.rowwise().mean();
			xmax_inliers = xmax_inliers.colwise() - mu;
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(xmax_inliers, Eigen::ComputeFullU | Eigen::ComputeFullV);
			y_dirs.push_back(svd.matrixU().col(0));
			y_dir_w.push_back(xmax_inliers.cols());
		}

		if (ymin_inliers.cols() > 2) {
			Eigen::Vector3f mu = ymin_inliers.rowwise().mean();
			ymin_inliers = ymin_inliers.colwise() - mu;
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(ymin_inliers, Eigen::ComputeFullU | Eigen::ComputeFullV);
			x_dirs.push_back(svd.matrixU().col(0));
			x_dir_w.push_back(ymin_inliers.cols());
		}

		if (ymax_inliers.cols() > 2) {
			Eigen::Vector3f mu = ymax_inliers.rowwise().mean();
			ymax_inliers = ymax_inliers.colwise() - mu;
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(ymax_inliers, Eigen::ComputeFullU | Eigen::ComputeFullV);
			x_dirs.push_back(svd.matrixU().col(0));
			x_dir_w.push_back(ymax_inliers.cols());
		}

		if (x_dirs.empty() && y_dirs.empty()) {
			x_dirs.push_back(R.row(0));
			y_dirs.push_back(R.row(1));
			x_dir_w.push_back(1);
			y_dir_w.push_back(1);
		}
		
		if (x_dirs.empty()) {
			x_dirs.push_back((y_dirs[0].cross(z_dir)).normalized());
			x_dir_w.push_back(y_dir_w[0]);
		}

		if (y_dirs.empty()) {
			y_dirs.push_back((z_dir.cross(x_dirs[0])).normalized());
			y_dir_w.push_back(x_dir_w[0]);
		}

		Eigen::Vector3f x_ref = x_dirs[0], y_ref = y_dirs[0];
		if ((x_ref.cross(y_ref)).dot(z_dir) < 0) {
			if (x_ref.dot(R.row(0)) < 0) {
				x_ref = -x_ref;
			} else {
				y_ref = -y_ref;
			}
		} else {
			if (x_ref.dot(R.row(0)) < 0) {
				x_ref = -x_ref;
				y_ref = -y_ref;
			}
		}

		for (int i = 0; i < x_dirs.size(); ++i) {
			if (x_dirs[i].dot(x_ref) < 0) {
				x_dirs[i] = -x_dirs[i];
			}
		}

		for (int i = 0; i < y_dirs.size(); ++i) {
			if (y_dirs[i].dot(y_ref) < 0) {
				y_dirs[i] = -y_dirs[i];
			}
		}

		// Convert all to x proposals
		std::vector<Eigen::Vector3f> x_proposals(0);
		std::vector<int> x_w(0);
		for (int i = 0; i < x_dirs.size(); ++i) {
			x_proposals.push_back(x_dirs[i]);
			x_w.push_back(x_dir_w[i]);
		}
		Eigen::Matrix3f R_z;
		R_z = Eigen::AngleAxisf(-M_PI/2.0, z_dir);
		for (int i = 0; i < y_dirs.size(); ++i) {
			x_proposals.push_back(R_z*y_dirs[i]);
			x_w.push_back(y_dir_w[i]);
		}

		// Weighted sum
		x_dir.setZero();
		for (int i = 0; i < x_proposals.size(); ++i) {
			x_dir = x_dir + x_w[i] * x_proposals[i];
		}
		x_dir.normalize();

		y_dir = (z_dir.cross(x_dir)).normalized();

		R.row(0) = x_dir;
		R.row(1) = y_dir;
		R.row(2) = z_dir;

		// int num_eq = x_dirs.size() + y_dirs.size() + 2;
		// Eigen::MatrixXf src(3, num_eq), dst(3, num_eq);
		// int j = 0;
		// for (int i = 0; i < x_dirs.size(); ++i) {
		// 	src.col(j) = Eigen::Vector3f::UnitX();
		// 	dst.col(j) = x_dirs[i];
		// 	j++;
		// }
		// for (int i = 0; i < y_dirs.size(); ++i) {
		// 	src.col(j) = Eigen::Vector3f::UnitY();
		// 	dst.col(j) = y_dirs[i];
		// 	j++;
		// }
		// src.col(j) = Eigen::Vector3f::UnitZ();
		// dst.col(j++) = z_dir;
		// src.col(j).setZero();
		// dst.col(j++).setZero();


		// Eigen::Vector3f t_tmp;
		// estimateRigidTransform3D(src, dst, R, t_tmp);
		// Eigen::Vector3f rot_axis = (z_dir.cross(R.row(2))).normalized();
		// float dp = z_dir.dot(R.row(2));
		// dp = (dp > 1.0) ? 1.0 : dp;
		// dp = (dp < -1.0) ? -1.0 : dp;
		// float rot_angle = std::acos(dp);
		// Eigen::Matrix3f R_corr;
		// R_corr = Eigen::AngleAxisf(rot_angle, rot_axis);
		// R = R*R_corr;

		// std::cout << "R_prev: " << R_prev.determinant() << std::endl;
		// std::cout << R_prev << std::endl;
		// std::cout << "R: " << R.determinant() << std::endl;
		// std::cout << R << std::endl;
		// std::cout << "norm:" << std::endl;
		// std::cout << (R - R_prev).norm() << std::endl;
		// std::cout << std::endl;

		// Check for convergence
		Eigen::Map<Eigen::Matrix3f>(Rv.data()) = R;
		if (((R - R_prev).norm() < tol) || (++iter >= max_iter) || (search_traj.find(Rv) != search_traj.end())){
			// if (search_traj.find(Rv) != search_traj.end()) std::cout << "CYCLE!" << std::endl;
			break;
		}
		search_traj.insert(Rv);
	}

	// std::cout << "Did " << iter << " iterations." << std::endl;

	Eigen::MatrixXf pts_3d_trans = R*pts_3d;
	Eigen::Vector3f min_pt_trans = pts_3d_trans.rowwise().minCoeff();
	min_pt_trans(2) = -sup_plane(3);
	Eigen::Vector3f max_pt_trans = pts_3d_trans.rowwise().maxCoeff();

	// Find side inliers
	Eigen::MatrixXf xmin_dist = pts_3d_trans.row(0).array() - min_pt_trans(0);
	Eigen::MatrixXf xmax_dist = max_pt_trans(0) - pts_3d_trans.row(0).array();
	Eigen::MatrixXf ymin_dist = pts_3d_trans.row(1).array() - min_pt_trans(1);
	Eigen::MatrixXf ymax_dist = max_pt_trans(1) - pts_3d_trans.row(1).array();

	Eigen::MatrixXf xmin_dist_inliers(1, (xmin_dist.array() <= inlier_dist).count());
	Eigen::MatrixXf xmax_dist_inliers(1, (xmax_dist.array() <= inlier_dist).count());
	Eigen::MatrixXf ymin_dist_inliers(1, (ymin_dist.array() <= inlier_dist).count());
	Eigen::MatrixXf ymax_dist_inliers(1, (ymax_dist.array() <= inlier_dist).count());

	int xmin_ind = 0, xmax_ind = 0, ymin_ind = 0, ymax_ind = 0;
	for (int i = 0; i < pts_3d.cols(); ++i) {
		if (xmin_dist(i) <= inlier_dist) xmin_dist_inliers(xmin_ind++) = xmin_dist(i);
		if (xmax_dist(i) <= inlier_dist) xmax_dist_inliers(xmax_ind++) = xmax_dist(i);
		if (ymin_dist(i) <= inlier_dist) ymin_dist_inliers(ymin_ind++) = ymin_dist(i);
		if (ymax_dist(i) <= inlier_dist) ymax_dist_inliers(ymax_ind++) = ymax_dist(i);
	}

	num_inliers = xmin_dist_inliers.cols() + xmax_dist_inliers.cols() + ymin_dist_inliers.cols() + ymax_dist_inliers.cols();
	mean_inlier_dist = 	xmin_dist_inliers.cols() * xmin_dist_inliers.mean() +
						xmax_dist_inliers.cols() * xmax_dist_inliers.mean() +
						ymin_dist_inliers.cols() * ymin_dist_inliers.mean() +
						ymax_dist_inliers.cols() * ymax_dist_inliers.mean();
	mean_inlier_dist /= num_inliers;

	// std::cout << "SCORES: " << num_inliers << ", " << mean_inlier_dist << std::endl;

	R.transposeInPlace();
	Eigen::Vector3f t = R*(max_pt_trans+min_pt_trans)/2.0;
	pose.linear() = R;
	pose.translation() = t;
	size = max_pt_trans-min_pt_trans;

	return true;
}

template<typename PointT>
bool fitMaximumContactBoundingBoxConstrained(const typename pcl::PointCloud<PointT>::ConstPtr &cloud, const Eigen::Vector4f &support_plane, Eigen::Affine3f &pose, Eigen::Vector3f &size, float inlier_dist, int num_samples = 18) {

	Eigen::Vector4f sup_plane = support_plane/support_plane.head(3).norm();
	Eigen::Vector3f normal = sup_plane.head(3);

	Eigen::Vector3f initial_dir = Eigen::Vector3f::UnitX();
	if (initial_dir.dot(normal) == 0.0) {
		initial_dir = Eigen::Vector3f::UnitY();
	}
	if (initial_dir.dot(normal) == 0.0) {
		initial_dir = Eigen::Vector3f::UnitZ();
	}
	initial_dir = (initial_dir - initial_dir.dot(normal)*normal).normalized();

	bool success_any = false;
	int num_inliers_best = 0;
	for (int i = 0; i < num_samples; ++i) {
		Eigen::Matrix3f normal_rot;
		normal_rot = Eigen::AngleAxisf(i*2*M_PI/num_samples, normal);
		Eigen::Vector3f initial_dir_curr = normal_rot*initial_dir;

		int num_inliers;
		float mean_inlier_dist;
		Eigen::Affine3f pose_curr;
		Eigen::Vector3f size_curr;
		bool success_curr = fitMaximumContactBoundingBoxConstrainedIterative<PointT>(cloud, support_plane, pose_curr, size_curr, num_inliers, mean_inlier_dist, initial_dir_curr, inlier_dist);

		success_any = success_any || success_curr;

		// std::cout << "SCORES: " << num_inliers << ", " << mean_inlier_dist << std::endl;

		if (num_inliers > num_inliers_best) {
			pose = pose_curr;
			size = size_curr;
			num_inliers_best = num_inliers;
		}
	}

	// std::cout << "SCORE: " << num_inliers_best << std::endl;

	return success_any;
}

#endif /* SISYPHUS_BOX_FITTING_HPP */
