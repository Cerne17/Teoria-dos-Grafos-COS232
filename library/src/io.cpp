#include "graph.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

std::unique_ptr<Graph> readGraph(const std::string &path, Representation rep) {
  std::unique_ptr<Graph> graph;
  std::ifstream in(path);

  if (!in) {
    throw std::runtime_error("could not open file " + path);
  }

  int n, a, b;

  if (!(in >> n))
    throw std::runtime_error("could not read vertex count 'n'");
  if (n < 0)
    throw std::runtime_error("vertex count can not be < 0. n = " +
                             std::to_string(n));

  if (rep == Representation::Matrix)
    graph = std::make_unique<AdjacencyMatrix>(n);
  else
    graph = std::make_unique<AdjacencyList>(n);

  while (in >> a >> b) {
    graph->addEdge(a, b);
  }

  graph->finalize();

  return graph;
}

void writeStats(const std::string &path, const std::string &graphName,
                const GraphStats &stats) {
  std::ofstream out(path, std::ios::app);

  if (!out)
    throw std::runtime_error("could not write file " + path);

  bool needHeader =
      !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  if (needHeader)
    out << "graphName, graphSize, graphEdges, minDegree, maxDegree, avgDegree, "
           "medianDegree, componentsAmount, largestComponentSize, "
           "smallestComponentSize, exactDiameter, approximateDiameter"
        << "\n";

  out << graphName << ", " << stats.n << ", " << stats.m << ", "
      << stats.minDegree << ", " << stats.maxDegree << ", " << stats.avgDegree
      << ", " << stats.medianDegree << ", " << stats.componentsAmount << ", "
      << stats.largestComponentSize << ", " << stats.smallestComponentSize
      << ", " << stats.exactDiameter << ", " << stats.approximateDiameter
      << "\n";
}

void writeSearchTree(const std::string &path, const std::string &graphName,
                     const SearchTree &tree) {
  std::ofstream out(path, std::ios::app);

  if (!out)
    throw std::runtime_error("could not write file " + path);

  bool needHeader =
      !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  if (needHeader)
    out << "graphName, node, parent, level" << "\n";

  for (int v = 1; v <= static_cast<int>(tree.parent.size()) - 1; ++v) {
    out << graphName << ", " << v << ", " << tree.parent[v] << ", "
        << tree.level[v] << "\n";
  }
}

void writeComponents(const std::string &path, const std::string &graphName,
                     const std::vector<std::vector<int>> &components) {
  std::ofstream out(path, std::ios::app);

  if (!out)
    throw std::runtime_error("could not write file " + path);

  bool needHeader =
      !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  if (needHeader)
    out << "graphName, node, componentId" << "\n";

  for (int i = 0; i < static_cast<int>(components.size()); i++) {
    for (auto v : components[i]) {
      out << graphName << ", " << v << ", " << i << "\n";
    }
  }
}
