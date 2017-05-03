#include <kabamaru/registration_utilities.hpp>

Eigen::Matrix4f homogeneousTransformationMatrixFromRt(const Eigen::Matrix3f &R, const Eigen::Vector3f &t) {
	Eigen::Matrix4f tform = Eigen::Matrix4f::Identity();
	tform.block<3,3>(0,0) = R;
	tform.block<3,1>(0,3) = t;
	return tform;
}

int rigidlyAlignPoseSets(const std::vector<Eigen::Matrix4f> &src_poses, const std::vector<Eigen::Matrix4f> &dst_poses, Eigen::Matrix4f &tform) {
	if (src_poses.size() != dst_poses.size()) return 0;
	Eigen::MatrixXf src_pts(3, 4*src_poses.size()), dst_pts(3, 4*dst_poses.size());
	for (int i = 0; i < src_poses.size(); ++i) {
		Eigen::Matrix3f R_src = src_poses[i].block<3,3>(0,0), R_dst = dst_poses[i].block<3,3>(0,0);
		Eigen::Vector3f t_src = src_poses[i].block<3,1>(0,3), t_dst = dst_poses[i].block<3,1>(0,3);
		for (int j = 0; j < 3; ++j) {
			src_pts.col(4*i + j) = t_src + R_src.col(j);
			dst_pts.col(4*i + j) = t_dst + R_dst.col(j);
		}
		src_pts.col(4*i + 3) = t_src;
		dst_pts.col(4*i + 3) = t_dst;
	}
	return estimateRigidTransform3D(src_pts, dst_pts, tform);
}

Eigen::VectorXf transferErrorRigid3D(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst, const Eigen::Matrix3f &R, const Eigen::Vector3f &t) {
	Eigen::Matrix3f Rinv = R.transpose();
	Eigen::MatrixXf src_t = (Rinv*dst).colwise() - Rinv*t;
	Eigen::MatrixXf dst_t = (R*src).colwise() + t;
	return (src-src_t).colwise().squaredNorm() + (dst-dst_t).colwise().squaredNorm();;
}

int estimateRigidTransform3D(const Eigen::MatrixXf &src_in, const Eigen::MatrixXf &dst_in, Eigen::Matrix3f &R, Eigen::Vector3f &t) {

	int N = src_in.cols();
	if (N < 3) {
		return 0;
	}

	Eigen::MatrixXf src = src_in;
	Eigen::MatrixXf dst = dst_in;

	Eigen::Vector3f mu_src, mu_dst;
	mu_src = src.rowwise().mean();
	mu_dst = dst.rowwise().mean();
	src = src.colwise() - mu_src;
	dst = dst.colwise() - mu_dst;
	Eigen::Matrix3f cov = dst*(src.transpose())/N;

	Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f U = svd.matrixU(), Vt = svd.matrixV().transpose();
	Eigen::Matrix3f tmp = U * Vt;
	Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
	if (tmp.determinant() < 0) {
		S(2,2) = -1;
	}

	R = U * S * Vt;
	t = mu_dst - R*mu_src;

	return 1;
}

int estimateRigidTransform3D(const Eigen::MatrixXf &src_in, const Eigen::MatrixXf &dst_in, Eigen::Matrix4f &tform) {
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	int success = estimateRigidTransform3D(src_in, dst_in, R, t);
	tform = homogeneousTransformationMatrixFromRt(R,t);
	return success;
}

int estimateRigidTransform3DRANSAC(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst, Eigen::Matrix3f &R, Eigen::Vector3f &t, std::vector<int> &inliers, int iter, int test_size, float dist_thresh, int inlier_thresh) {

	int N = src.cols();
	if (N < test_size) {
		return 0;
	}

	// int inlier_thresh = inlier_ratio*N;
	inlier_thresh = (inlier_thresh > test_size) ? inlier_thresh : test_size;
	inlier_thresh = (inlier_thresh > N) ? N : inlier_thresh;

	std::vector<int> perm(N);
	for (int i = 0; i < N; i++) {
		perm[i] = i;
	}

	int k = 0, i, j;
	int N_best = 0, N_test;
	Eigen::Matrix3f R_tmp;
	Eigen::Vector3f t_tmp;
	Eigen::MatrixXf test_src, test_dst;
	Eigen::VectorXf dist_tmp;
	Eigen::Matrix<bool, Eigen::Dynamic, 1> comp_tmp;
	while (k++ < iter) {
		test_src = Eigen::MatrixXf(3,test_size);
		test_dst = Eigen::MatrixXf(3,test_size);

		std::random_shuffle(perm.begin(), perm.end());
		for (int i = 0; i < test_size; i++) {
			test_src.col(i) = src.col(perm[i]);
			test_dst.col(i) = dst.col(perm[i]);
		}

		estimateRigidTransform3D(test_src,test_dst,R_tmp,t_tmp);

		dist_tmp = transferErrorRigid3D(src,dst,R_tmp,t_tmp);
		comp_tmp = dist_tmp.array() < dist_thresh;
		N_test = comp_tmp.count();

		if (N_test < test_size) {
			continue;
		}

		std::vector<int> test_inliers(N_test);
		test_src = Eigen::MatrixXf(3,N_test);
		test_dst = Eigen::MatrixXf(3,N_test);
		j = 0;
		for (int i = 0; i < N; i++) {
			if (comp_tmp(i)) {
				test_inliers[j] = i;
				test_src.col(j) = src.col(i);
				test_dst.col(j) = dst.col(i);
				j++;
			}
		}

		// Reestimate
		estimateRigidTransform3D(test_src,test_dst,R_tmp,t_tmp);

		if (N_test >= inlier_thresh) {
			N_best = N_test;
			R = R_tmp;
			t = t_tmp;
			inliers = test_inliers;
			break;
		}

		if (N_test > N_best) {
			N_best = N_test;
			R = R_tmp;
			t = t_tmp;
			inliers = test_inliers;
		}
	}

	// std::cout << "iter: " << k << std::endl;
	return N_best > 0;
}

int estimateRigidTransform3DRANSAC(const Eigen::MatrixXf &src, const Eigen::MatrixXf &dst, Eigen::Matrix4f &tform, std::vector<int>& inliers, int iter, int test_size, float dist_thresh, int inlier_thresh) {
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	int success = estimateRigidTransform3DRANSAC(src, dst, R, t, inliers, iter, test_size, dist_thresh, inlier_thresh);
	tform = homogeneousTransformationMatrixFromRt(R,t);
	return success;
}

Eigen::Vector3f projectiveToRealWorld(const Eigen::Vector3f &pt_pr, const Eigen::Matrix3f &K) {
	Eigen::Vector3f tmp;
	tmp(0) = (pt_pr(0) - K(0,2)) * pt_pr(2) / K(0,0);
	tmp(1) = (pt_pr(1) - K(1,2)) * pt_pr(2) / K(1,1);
	tmp(2) = pt_pr(2);
	return tmp;
}

Eigen::Vector3f realWorldToProjective(const Eigen::Vector3f &pt_rw, const Eigen::Matrix3f &K) {
	Eigen::Vector3f tmp;
	tmp(0) = pt_rw(0) * K(0,0) / pt_rw(2) + K(0,2);
	tmp(1) = pt_rw(1) * K(1,1) / pt_rw(2) + K(1,2);
	tmp(2) = pt_rw(2);
	return tmp;
}

void transformConvexPolytope(std::vector<Eigen::Vector4f> &polytope, const Eigen::Matrix4f &tform) {
	Eigen::Matrix4f tform_it = tform.inverse().transpose();
	for (int i = 0; i < 4; i++) {
		polytope[i] = tform_it * polytope[i];
	}
}

Eigen::MatrixXf extractMatrixColumnsFromIndices(const Eigen::MatrixXf &mat, const std::vector<int> &ind) {
	Eigen::MatrixXf res(mat.rows(),ind.size());
	for (int i = 0; i < ind.size(); i++) {
		res.col(i) = mat.col(ind[i]);
	}
	return res;
}

std::vector<float> extractSIFTDescriptorsFromIndices(const std::vector<float> &descr, const std::vector<int> &ind) {
	std::vector<float> res(128*ind.size());
	for (int i = 0; i < ind.size(); i++) {
		for (int j = 0; j < 128; j++) {
			res[128*i+j] = descr[128*ind[i]+j];
		}
	}
	return res;
}

Eigen::MatrixXf extractSIFTKeypoint2DCoordinates(const std::vector<SiftGPU::SiftKeypoint> &keys) {
	int num = keys.size();
	Eigen::MatrixXf res(2,num);
	for (int i = 0; i < num; ++i) {
		res(0,i) = keys[i].x;
		res(1,i) = keys[i].y;
	}
	return res;
}

int estimateRigidTransformSIFT3DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix3f &R, Eigen::Vector3f &t, std::vector<int> &inliers) {
	std::vector<int> frame_ind, model_ind;
	sift_engine->getCorrespondences(frame_descr, model_descr, frame_ind, model_ind);
	Eigen::MatrixXf frame_loc_tmp = extractMatrixColumnsFromIndices(frame_loc, frame_ind);
	Eigen::MatrixXf model_loc_tmp = extractMatrixColumnsFromIndices(model_loc, model_ind);

	inliers.clear();
	int success = estimateRigidTransform3DRANSAC(frame_loc_tmp, model_loc_tmp, R, t, inliers, iter, 4, dist_thresh, min_inlier_count);

	return (success > 0) && (inliers.size() > 10);
}

int estimateRigidTransformSIFT3DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix4f &tform, std::vector<int> &inliers) {
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	int success = estimateRigidTransformSIFT3DTo3DRANSAC(frame_loc, frame_descr, model_loc, model_descr, sift_engine, iter, dist_thresh, min_inlier_count, R, t, inliers);
	tform = homogeneousTransformationMatrixFromRt(R,t);
	return success;
}

int estimateRigidTransformSIFT2DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, const cv::Mat &K, const cv::Mat &d, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix3f &R, Eigen::Vector3f &t, std::vector<int> &inliers) {

	std::vector<int> frame_ind, model_ind;
	sift_engine->getCorrespondences(frame_descr, model_descr, frame_ind, model_ind);

	if (frame_ind.size() < 4) {
		return 0;
	}

	Eigen::MatrixXf frame_loc_tmp = extractMatrixColumnsFromIndices(frame_loc, frame_ind).transpose();
	Eigen::MatrixXf model_loc_tmp = extractMatrixColumnsFromIndices(model_loc, model_ind).transpose();

	cv::Mat model_loc_tmp_cv, frame_loc_tmp_cv;
	cv::eigen2cv(model_loc_tmp, model_loc_tmp_cv);
	cv::eigen2cv(frame_loc_tmp, frame_loc_tmp_cv);

	double confidence = static_cast<double>(min_inlier_count)/static_cast<double>(frame_ind.size());
	if (confidence >= 1) confidence = 0.99;

	inliers.clear();
	cv::Mat R_cv, t_cv;
	cv::solvePnPRansac(model_loc_tmp_cv, frame_loc_tmp_cv, K, d, R_cv, t_cv, false, iter, dist_thresh, confidence, inliers, cv::SOLVEPNP_ITERATIVE);
	cv::Rodrigues(R_cv, R_cv);

	Eigen::Matrix3f R_tmp;
	Eigen::Vector3f t_tmp;
	cv::cv2eigen(R_cv, R_tmp);
	cv::cv2eigen(t_cv, t_tmp);

	R = R_tmp.transpose();
	t = -R*t_tmp;

	return inliers.size() > 10;
}

int estimateRigidTransformSIFT2DTo3DRANSAC(const Eigen::MatrixXf &frame_loc, const std::vector<float> &frame_descr, const Eigen::MatrixXf &model_loc, const std::vector<float> &model_descr, const cv::Mat &K, const cv::Mat &d, SIFTEngine *sift_engine, int iter, float dist_thresh, int min_inlier_count, Eigen::Matrix4f &tform, std::vector<int> &inliers) {
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	int success = estimateRigidTransformSIFT2DTo3DRANSAC(frame_loc, frame_descr, model_loc, model_descr, K, d, sift_engine, iter, dist_thresh, min_inlier_count, R, t, inliers);
	tform = homogeneousTransformationMatrixFromRt(R,t);
	return success;
}
