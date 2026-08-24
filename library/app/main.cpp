#include "graph.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main() {
  // std::string path_in = "data/graph.txt";
  std::string path_in = "data/multiple_components_graph.txt";
  std::string path_out_stats = "data/stats.csv";
  std::string path_out_bfs_tree = "data/bfs_tree.csv";
  std::string path_out_dfs_tree = "data/dfs_tree.csv";
  std::string repr;

  std::filesystem::remove(path_out_stats);
  std::filesystem::remove(path_out_bfs_tree);
  std::filesystem::remove(path_out_dfs_tree);

  try {
    for (Representation r : {Representation::List, Representation::Matrix}) {

      if (r == Representation::List)
        repr = "list";
      else
        repr = "matrix";

      std::unique_ptr<Graph> g = readGraph(path_in, Representation::List);
      GraphStats stats = computeStats(*g);
      SearchTree bfsTree = bfs(*g, 1);
      SearchTree dfsTree = dfs(*g, 1);

      writeStats(path_out_stats, repr, stats);
      writeSearchTree(path_out_bfs_tree, repr, bfsTree);
      writeSearchTree(path_out_dfs_tree, repr, dfsTree);

      std::vector<SearchTree> components = getComponents(*g);

      int k = 1;
      std::cout << "EXAMPLE COMPONENTS FOR REPR. " << repr << "\n";
      for (auto component : components) {
        std::cout << "Component #" << k << ":\n";
        std::cout << "NODE | PARENT | LEVEL" << "\n";
        for (int i = 1; i <= g->n; i++) {
          std::cout << i << "    | " << component.parent[i] << "      | "
                    << component.level[i] << "\n";
        }
        k++;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
