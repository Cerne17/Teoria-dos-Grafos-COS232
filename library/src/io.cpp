#include "graph.hpp"

#include <fstream>
#include <stdexcept>

Graph readGraph(const std::string &path) {
  std::ifstream in(path);

  if (!in) {
    throw std::runtime_error("could not open file " + path);
  }

  int n, a, b;
  in >> n;

  if (!n)
    throw std::runtime_error("could not read vertex count n = " +
                             std::to_string(n));
  if (n < 0)
    throw std::runtime_error("vertex count can not be < 0. n = " +
                             std::to_string(n));

  Graph graph(n);
  while (in >> a >> b) {
    graph.addEdge(a, b);
  }

  return graph;
}
