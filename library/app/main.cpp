#include "graph.hpp"

#include <iostream>

int main() {
  std::string path = "data/graph.txt";

  try {
    Graph g = readGraph(path);
    std::cout << "Graph created\n"
              << "vertex Count n = " << g.n << "\n"
              << "edge Count m = " << g.m << std::endl;

    GraphStats stats = computeStats(g);

    std::cout << "min degree: " << stats.minDegree
              << "\nmax degree: " << stats.maxDegree
              << "\navg degree: " << stats.avgDegree
              << "\nmedian degree: " << stats.medianDegree << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
