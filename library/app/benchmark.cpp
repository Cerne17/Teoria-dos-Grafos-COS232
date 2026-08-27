#include "graph.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

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
  if (argc < 4) {
    std::cerr << "usage: benchmark <graphfile> <list|matrix> <time|mem>\n";
    return 1;
  }

  std::string graphFile = argv[1];
  std::string repArg = argv[2];
  std::string mode = argv[3];

  Representation rep;
  if (repArg == "matrix") {
    rep = Representation::Matrix;
  } else if (repArg == "list") {
    rep = Representation::List;
  } else {
    std::cerr << "Invalid representation: " << repArg << "\n";
    return 1;
  }

  std::string graphName = cleanGraphName(graphFile);

  try {
    std::unique_ptr<Graph> g = readGraph(graphFile, rep);

    if (mode == "time") {
      long long totalMicros = 0;
      long long sink = 0;

      for (int i = 0; i < 100; i++) {
        int start = (i % g->n) + 1;

        auto t0 = std::chrono::steady_clock::now();
        SearchTree tree = bfs(*g, start);
        auto t1 = std::chrono::steady_clock::now();

        totalMicros +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        sink += tree.level[g->n];
      }
      double avgBfs = static_cast<double>(totalMicros) / 100.0;
      std::cout << graphName << "," << repArg << ",bfs," << avgBfs << "\n";
      std::cerr << "sink=" << sink << "\n";

      totalMicros = 0;
      sink = 0;

      for (int i = 0; i < 100; i++) {
        int start = (i % g->n) + 1;

        auto t0 = std::chrono::steady_clock::now();
        SearchTree tree = dfs(*g, start);
        auto t1 = std::chrono::steady_clock::now();

        totalMicros +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        sink += tree.level[g->n];
      }
      double avgDfs = static_cast<double>(totalMicros) / 100.0;
      std::cout << graphName << "," << repArg << ",dfs," << avgDfs << "\n";
      std::cerr << "sink=" << sink << "\n";

    } else if (mode == "mem") {
      std::cout << "READY\n";
      std::cout.flush();
      std::string dummy;
      std::getline(std::cin, dummy);

    } else {
      std::cerr << "unknown mode: " << mode << "\n";
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
