#include "graph.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// Strip directory and extension: "data/grafo_1.txt" -> "grafo_1"
static std::string cleanGraphName(const std::string &path) {
  std::string name = path;
  size_t slash = name.find_last_of("/\\");
  if (slash != std::string::npos)
    name = name.substr(slash + 1);
  size_t dot = name.find_last_of('.');
  if (dot != std::string::npos)
    name = name.substr(0, dot);
  return name;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: graphs <graphfile> [list|matrix|both]\n";
    return 1;
  }

  std::string path_in = argv[1];
  std::string repSpec = (argc > 2) ? argv[2] : "both";
  std::string name = cleanGraphName(path_in);

  // Which representations to generate output files for.
  std::vector<Representation> reps;
  if (repSpec == "list") {
    reps = {Representation::List};
  } else if (repSpec == "matrix") {
    reps = {Representation::Matrix};
  } else if (repSpec == "both") {
    reps = {Representation::List, Representation::Matrix};
  } else {
    std::cerr << "Invalid representation: " << repSpec << "\n";
    return 1;
  }

  std::string path_out_stats = "data/stats_" + name + ".csv";
  std::string path_out_bfs_tree = "data/bfs_tree_" + name + ".csv";
  std::string path_out_dfs_tree = "data/dfs_tree_" + name + ".csv";
  std::string path_out_components = "data/components_" + name + ".csv";

  std::filesystem::remove(path_out_stats);
  std::filesystem::remove(path_out_bfs_tree);
  std::filesystem::remove(path_out_dfs_tree);
  std::filesystem::remove(path_out_components);

  try {
    for (Representation r : reps) {
      std::string repr = (r == Representation::List) ? "list" : "matrix";

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

      std::cout << "Done for " << name << " (" << repr << ")\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
