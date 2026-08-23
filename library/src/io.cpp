#include "graph.hpp"

#include <fstream>
#include <stdexcept>

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
