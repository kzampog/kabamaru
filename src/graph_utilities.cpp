#include <stack>
#include <algorithm>

#include <kabamaru/graph_utilities.hpp>

std::vector<std::vector<int> > adjacencyListFromAdjacencyMatrix(const std::vector<std::vector<char> > &adj_mat) {
	std::vector<std::vector<int> > adj_list;
	for (int i = 0; i < adj_mat.size(); ++i) {
		std::vector<int> nb_curr;
		for (int j = 0; j < adj_mat[i].size(); ++j) {
			if (adj_mat[i][j] == 1) {
				nb_curr.push_back(j);
			}
		}
		adj_list.push_back(nb_curr);
	}

	// for (int i = 0; i < adj_list.size(); ++i) {
	// 	std::cout << i << ":";
	// 	for (int j = 0; j < adj_list[i].size(); ++j) {
	// 		std::cout << " " << adj_list[i][j];
	// 	}
	// 	std::cout << std::endl;
	// }

	return adj_list;
}


std::vector<std::vector<int> > findConnectedComponentsDFS(const std::vector<std::vector<int> > &adj_list) {

	std::vector<std::vector<int> > cc;
	std::vector<char> visited(adj_list.size(), 0);

	for (int i = 0; i < adj_list.size(); ++i) {
		std::vector<int> cc_curr;
		std::stack<int> pending;
		pending.push(i);

		while (!pending.empty()) {
			int n = pending.top();
			pending.pop();
			if (visited[n] == 0) {
				visited[n] = 1;
				cc_curr.push_back(n);
				for (int j = 0; j < adj_list[n].size(); ++j) {
					pending.push(adj_list[n][j]);
				}
			}
		}

		if (!cc_curr.empty()) {
			cc.push_back(cc_curr);
		}
	}

	return cc;
}

std::vector<std::vector<int> > findConnectedComponentsDFS(const std::vector<std::vector<char> > &adj_mat) {
	return findConnectedComponentsDFS(adjacencyListFromAdjacencyMatrix(adj_mat));
}

void generateAllConnectedSubsetsAux(const std::vector<std::vector<int> > &adj_list, std::set<int> path, int last, std::set<std::set<int> > &subsets) {

	if (subsets.empty()) {
		for (int i = 0; i < adj_list.size(); ++i) {
			std::set<int> path_tmp;
			path_tmp.insert(i);
			subsets.insert(path_tmp);
			generateAllConnectedSubsetsAux(adj_list, path_tmp, i, subsets);
		}
		return;
	}

	for (int i = 0; i < adj_list[last].size(); ++i) {
		if (path.find(adj_list[last][i]) == path.end()) {
			std::set<int> path_tmp = path;
			path_tmp.insert(adj_list[last][i]);
			subsets.insert(path_tmp);
			generateAllConnectedSubsetsAux(adj_list, path_tmp, adj_list[last][i], subsets);
		}
	}
}

std::set<std::set<int> > generateAllConnectedSubsets(const std::vector<std::vector<int> > &adj_list) {
	std::set<std::set<int> > subsets;
	std::set<int> path;
	generateAllConnectedSubsetsAux(adj_list, path, -1, subsets);
	return subsets;
}

bool cliqueCheck(const std::vector<std::vector<char> > &adj_mat, const std::vector<int> &ss) {
	for (int i = 0; i < ss.size(); ++i) {
		for (int j = i + 1; j < ss.size(); ++j) {
			if (adj_mat[ss[i]][ss[j]] == 0) {
				return false;
			}
		}
	}
	return true;
}

std::vector<std::vector<int> > findCliques(const std::vector<std::vector<char> > &adj_mat) {
	std::set<std::set<int> > subsets = generateAllConnectedSubsets(adjacencyListFromAdjacencyMatrix(adj_mat));

	// for (std::set<std::set<int> >::iterator set = subsets.begin(); set != subsets.end();) {
	// 	std::set<std::set<int> >::iterator current = set++;
	// 	if (current->size() < 2) {
	// 		subsets.erase(current);
	// 	}
	// }

	// std::cout << "Connected subsets:" << std::endl;
	// for (std::set<std::set<int> >::iterator ss = subsets.begin(); ss != subsets.end(); ++ss) {
	// 	for (std::set<int>::iterator e = ss->begin(); e != ss->end(); ++e) {
	// 		std::cout << *e << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	std::vector<std::vector<int> > cliques;
	while (!subsets.empty()) {
		int max_size = 0;
		std::set<std::set<int> >::iterator max_it;
		for (std::set<std::set<int> >::iterator ss = subsets.begin(); ss != subsets.end(); ++ss) {
			if (ss->size() > max_size) {
				max_size = ss->size();
				max_it = ss;
			}
		}

		std::set<int> set_tmp = *max_it;
		std::vector<int> vec_tmp(set_tmp.begin(),set_tmp.end());
		subsets.erase(max_it);
		if (cliqueCheck(adj_mat, vec_tmp)) {
			cliques.push_back(vec_tmp);
			// update subsets
			for (std::set<std::set<int> >::iterator ss = subsets.begin(); ss != subsets.end();) {
				std::set<std::set<int> >::iterator current = ss++;
				std::vector<int> intersection;
				std::set_intersection(current->begin(), current->end(), set_tmp.begin(), set_tmp.end(), std::back_inserter(intersection));
				if (intersection.size() > 0) {
					subsets.erase(current);
				}
			}
		}
	}

	// std::cout << "Cliques:" << std::endl;
	// for (int i = 0; i < cliques.size(); ++i) {
	// 	for (int j = 0; j < cliques[i].size(); ++j) {
	// 		std::cout << " " << cliques[i][j];
	// 	}
	// 	std::cout << std::endl;
	// }

	return cliques;
}
