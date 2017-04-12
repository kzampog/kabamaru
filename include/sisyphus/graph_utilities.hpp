#ifndef GRAPH_UTILITIES_HPP
#define GRAPH_UTILITIES_HPP

#include <vector>
#include <set>

std::vector<std::vector<int> > adjacencyListFromAdjacencyMatrix(const std::vector<std::vector<char> > &adj_mat);

std::vector<std::vector<int> > findConnectedComponentsDFS(const std::vector<std::vector<int> > &adj_list);

std::vector<std::vector<int> > findConnectedComponentsDFS(const std::vector<std::vector<char> > &adj_mat);

void generateAllConnectedSubsetsAux(const std::vector<std::vector<int> > &adj_list, std::set<int> path, int last, std::set<std::set<int> > &subsets);

std::set<std::set<int> > generateAllConnectedSubsets(const std::vector<std::vector<int> > &adj_list);

bool cliqueCheck(const std::vector<std::vector<char> > &adj_mat, const std::vector<int> &ss);

std::vector<std::vector<int> > findCliques(const std::vector<std::vector<char> > &adj_mat);

#endif /* GRAPH_UTILITIES_HPP */
