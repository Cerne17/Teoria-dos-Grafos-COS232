#include "graph.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Same file the Python harness reads, so both drivers stay in sync.
static const std::string CONFIG_PATH = "config.csv";

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

// Drop surrounding whitespace so "grafo_1 , both" parses like "grafo_1,both".
static std::string trim(const std::string &s) {
  size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return "";
  size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

// Which representations to generate output files for.
static std::vector<Representation> representationsFor(const std::string &spec) {
  if (spec == "list")
    return {Representation::List};
  if (spec == "matrix")
    return {Representation::Matrix};
  if (spec == "both")
    return {Representation::List, Representation::Matrix};
  throw std::runtime_error("invalid representation: " + spec);
}

// Reads the graph/representation pairs from config.csv. The header row is
// skipped when present, and blank lines are ignored.
static std::vector<std::pair<std::string, std::string>>
readConfig(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("could not open " + path);

  std::vector<std::pair<std::string, std::string>> rows;
  std::string line;

  while (std::getline(in, line)) {
    size_t comma = line.find(',');
    if (comma == std::string::npos)
      continue;

    std::string name = trim(line.substr(0, comma));
    std::string rep = trim(line.substr(comma + 1));
    if (name.empty() || rep.empty())
      continue;
    if (rows.empty() && name == "graph") // header
      continue;

    rows.push_back({name, rep});
  }

  if (rows.empty())
    throw std::runtime_error("no graphs listed in " + path);

  return rows;
}

// Runs the full pipeline for one graph and rewrites its four output CSVs.
static void processGraph(const std::string &path_in,
                         const std::string &repSpec) {
  std::vector<Representation> reps = representationsFor(repSpec);
  std::string name = cleanGraphName(path_in);

  std::string path_out_stats = "data/stats_" + name + ".csv";
  std::string path_out_bfs_tree = "data/bfs_tree_" + name + ".csv";
  std::string path_out_dfs_tree = "data/dfs_tree_" + name + ".csv";
  std::string path_out_components = "data/components_" + name + ".csv";

  std::filesystem::remove(path_out_stats);
  std::filesystem::remove(path_out_bfs_tree);
  std::filesystem::remove(path_out_dfs_tree);
  std::filesystem::remove(path_out_components);

  for (Representation r : reps) {
    std::string repr = (r == Representation::List) ? "list" : "matrix";

    std::unique_ptr<Graph> g = readGraph(path_in, r);
    GraphStats stats = computeStats(*g, kExactDiameterBudget);
    SearchTree bfsTree = bfs(*g, 1);
    SearchTree dfsTree = dfs(*g, 1);
    std::vector<std::vector<int>> breadthComponents = getComponentsByBreadth(*g);
    std::vector<std::vector<int>> depthComponents = getComponentsByDepth(*g);

    writeStats(path_out_stats, repr, stats);
    writeSearchTree(path_out_bfs_tree, repr, bfsTree);
    writeSearchTree(path_out_dfs_tree, repr, dfsTree);
    writeComponents(path_out_components, repr + " breadth", breadthComponents);
    writeComponents(path_out_components, repr + " depth", depthComponents);

    if (stats.exactDiameter < 0)
      std::cout << "  exact diameter over budget, approximation only\n";
    std::cout << "Done for " << name << " (" << repr << ")\n";
  }
}

int main(int argc, char *argv[]) {
  // With arguments, process just the graph named on the command line. With
  // none, walk config.csv the same way the Python harness does.
  std::vector<std::pair<std::string, std::string>> jobs;

  if (argc >= 2) {
    jobs.push_back({argv[1], (argc > 2) ? argv[2] : "both"});
  } else {
    try {
      for (const auto &row : readConfig(CONFIG_PATH))
        jobs.push_back({"data/" + row.first + ".txt", row.second});
    } catch (const std::exception &e) {
      std::cerr << "error: " << e.what() << "\n"
                << "usage: graphs [<graphfile> [list|matrix|both]]\n"
                << "  with no arguments, every row of " << CONFIG_PATH
                << " is processed\n";
      return 1;
    }
    std::cout << "no arguments given, processing " << jobs.size()
              << " graph(s) from " << CONFIG_PATH << "\n";
  }

  // Keep going after a failure so one unreadable graph does not cost the rest,
  // the same way the Python harness does.
  int failures = 0;
  for (const auto &job : jobs) {
    try {
      processGraph(job.first, job.second);
    } catch (const std::exception &e) {
      std::cerr << "  FAILED for " << job.first << ": " << e.what() << "\n";
      failures++;
    }
  }

  if (failures > 0)
    std::cerr << failures << " of " << jobs.size() << " failed\n";

  return failures == 0 ? 0 : 1;
}
