#include <sisyphus/box_detection.hpp>

std::vector<Eigen::Vector3f> extractNormalsWithOrthogonalCounterpart(const std::vector<Eigen::Vector3f> &normals, float angle_thresh) {
	int n = normals.size();
	std::vector<std::vector<char> > adj_mat(n);
	for (int i = 0; i < adj_mat.size(); ++i) {
		adj_mat[i] = std::vector<char>(n);
	}
	for (int i = 0; i < adj_mat.size(); ++i) {
		for (int j = i+1; j < adj_mat[i].size(); ++j) {
			adj_mat[i][j] = std::abs(normals[i].dot(normals[j])) < std::sin(angle_thresh);
			adj_mat[j][i] = adj_mat[i][j];
		}
	}

	std::vector<Eigen::Vector3f> res;
	for (int i = 0; i < adj_mat.size(); ++i) {
		for (int j = 0; j < adj_mat[i].size(); ++j) {
			if (adj_mat[i][j] == 1) {
				res.push_back(normals[i]);
				break;
			}
		}
	}

	// for (int i = 0; i < adj_mat.size(); ++i) {
	// 	for (int j = 0; j < adj_mat[i].size(); ++j) {
	// 		std::cout << (int)adj_mat[i][j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << res.size() << std::endl;

	return res;
}

Eigen::Matrix3f getBoxAxesFromSideNormals(const std::vector<Eigen::Vector3f> &side_normals, float angle_thresh) {
	std::vector<Eigen::Vector3f> dir_map(3);
	Eigen::Matrix3f tmp = Eigen::Matrix3f::Identity();
	for (int i = 0; i < 3; ++i) {
		dir_map[i] = tmp.col((int)i);
	}

	std::vector<Eigen::Vector3f> normals;
	std::vector<int> dir_ids;
	int max_id = -1;
	Eigen::Vector3f third_dir_ref;

	for (int i = 0; i < side_normals.size(); ++i) {
		bool found = false;
		Eigen::Vector3f normal = side_normals[i];
		if (normals.empty()) {
			normals.push_back(normal);
			dir_ids.push_back(++max_id);
			continue;
		}
		for (int j = 0; j < normals.size(); ++j) {
			float dot_prod = normal.dot(normals[j]);
			if (std::abs(dot_prod) > std::cos(angle_thresh)) {
				if (dot_prod < 0) {
					normal = -normal;
				}
				normals.push_back(normal);
				dir_ids.push_back(dir_ids[j]);
				found = true;
				break;
			}
		}
		if (!found) {
			if (max_id == 2) {
				continue;
			}
			if (max_id == 0) {
				third_dir_ref = normals[0].cross(normal).normalized();
			}
			if (max_id == 1 && normal.dot(third_dir_ref) < 0) {
				normal = -normal;
			}
			normals.push_back(normal);
			dir_ids.push_back(++max_id);
		}
	}

	if (normals.size() < 3) {
		normals.push_back(third_dir_ref);
		dir_ids.push_back(2);
	}

	// for (int i = 0; i < normals.size(); ++i) {
	// 	std::cout << normals[i].transpose() << "      " << dir_ids[i] << std::endl;
	// }

	Eigen::MatrixXf src(3,normals.size()+1), dst(3,normals.size()+1);
	for (int i = 0; i < normals.size(); ++i) {
		src.col(i) = dir_map[dir_ids[i]];
		dst.col(i) = normals[i];
	}
	src.col(normals.size()).setZero();
	dst.col(normals.size()).setZero();

	// std::cout << "SRC:" << std::endl;
	// std::cout << src << std::endl;
	// std::cout << "DST:" << std::endl;
	// std::cout << dst << std::endl;

	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	estimateRigidTransform3D(src, dst, R, t);
	// std::cout << "R:" << std::endl;
	// std::cout << R << std::endl;
	// std::cout << "t:" << std::endl;
	// std::cout << t.transpose() << std::endl;

	return R;
}
