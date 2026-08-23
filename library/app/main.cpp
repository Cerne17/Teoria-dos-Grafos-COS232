#include "graph.hpp"

#include <iostream>
#include <string>

int main() {
  std::string path = "data/graph.txt";

  try {
    for (Representation r : {Representation::List, Representation::Matrix}) {

      if (r == Representation::List)
        std::cout << "REPRESENTATION - LIST" << std::endl;
      else
        std::cout << "REPRESENTATION - MATRIX" << std::endl;

      std::unique_ptr<Graph> g = readGraph(path, Representation::List);
      std::cout << "Graph created\n"
                << "vertex Count n = " << g->n << "\n"
                << "edge Count m = " << g->m << std::endl;

      GraphStats stats = computeStats(*g);

      std::cout << "min degree: " << stats.minDegree
                << "\nmax degree: " << stats.maxDegree
                << "\navg degree: " << stats.avgDegree
                << "\nmedian degree: " << stats.medianDegree << std::endl;

      SearchTree bfsTree = bfs(*g, 1);

      std::cout << "BFS" << std::endl;

      for (int i = 1; i <= g->n; i++)
        std::cout << "parent of " << i << ": " << bfsTree.parent[i]
                  << std::endl;

      for (int i = 1; i <= g->n; i++)
        std::cout << "level of " << i << ": " << bfsTree.level[i] << std::endl;

      SearchTree dfsTree = dfs(*g, 1);

      std::cout << "DFS" << std::endl;

      for (int i = 1; i <= g->n; i++)
        std::cout << "parent of " << i << ": " << dfsTree.parent[i]
                  << std::endl;

      for (int i = 1; i <= g->n; i++)
        std::cout << "level of " << i << ": " << dfsTree.level[i] << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
