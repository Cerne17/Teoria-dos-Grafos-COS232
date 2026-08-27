#include "graph.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::string path_in = (argc > 1) ? argv[1] : "data/grafo_1.txt";
  std::string path_out_stats = "data/stats.csv";
  std::string path_out_bfs_tree = "data/bfs_tree.csv";
  std::string path_out_dfs_tree = "data/dfs_tree.csv";
  std::string repr;
  std::string path_out_components = "data/components.csv";

  std::filesystem::remove(path_out_stats);
  std::filesystem::remove(path_out_bfs_tree);
  std::filesystem::remove(path_out_dfs_tree);
  std::filesystem::remove(path_out_components);

  try {
    for (Representation r : {Representation::List, Representation::Matrix}) {

      if (r == Representation::List)
        repr = "list";
      else
        repr = "matrix";

      std::unique_ptr<Graph> g = readGraph(path_in, r);
      GraphStats stats = computeStats(*g);
      SearchTree bfsTree = bfs(*g, 1);
      SearchTree dfsTree = dfs(*g, 1);
      std::vector<std::vector<int>> breadthComponents =
          getComponentsByBreadth(*g);
      std::vector<std::vector<int>> depthComponents = getComponentsByDepth(*g);

      writeStats(path_out_stats, repr, stats);
      writeSearchTree(path_out_bfs_tree, repr, bfsTree);
      writeSearchTree(path_out_dfs_tree, repr, dfsTree);
      writeComponents(path_out_components, repr + " breadth",
                      breadthComponents);
      writeComponents(path_out_components, repr + " depth", depthComponents);

      std::cout << "Done for " << repr << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
