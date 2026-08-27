#include "graph.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
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

// --- Timing: 100 BFS + 100 DFS, self-timed, algorithm-only. ---
static void runTiming(const Graph &g, const std::string &graphName,
                      const std::string &repArg) {
  long long totalMicros = 0;
  long long sink = 0;

  for (int i = 0; i < 100; i++) {
    int start = (i % g.n) + 1;

    auto t0 = std::chrono::steady_clock::now();
    SearchTree tree = bfs(g, start);
    auto t1 = std::chrono::steady_clock::now();

    totalMicros +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    sink += tree.level[g.n];
  }
  double avgBfs = static_cast<double>(totalMicros) / 100.0;
  std::cout << graphName << "," << repArg << ",bfs," << avgBfs << "\n";
  std::cerr << "sink=" << sink << "\n";

  totalMicros = 0;
  sink = 0;

  for (int i = 0; i < 100; i++) {
    int start = (i % g.n) + 1;

    auto t0 = std::chrono::steady_clock::now();
    SearchTree tree = dfs(g, start);
    auto t1 = std::chrono::steady_clock::now();

    totalMicros +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    sink += tree.level[g.n];
  }
  double avgDfs = static_cast<double>(totalMicros) / 100.0;
  std::cout << graphName << "," << repArg << ",dfs," << avgDfs << "\n";
  std::cerr << "sink=" << sink << "\n";
}

// --- Report: specific case-study answers (Q4-Q7). ---
static void runReport(const Graph &g, const std::string &graphName,
                      const std::string &repArg) {
  std::vector<int> sources = {1, 2, 3};
  std::vector<int> targets = {10, 20, 30};

  // Q4: parents of targets in BFS/DFS trees rooted at each source.
  for (int s : sources) {
    if (s > g.n)
      continue;
    SearchTree bt = bfs(g, s);
    SearchTree dt = dfs(g, s);
    for (int t : targets) {
      if (t > g.n)
        continue;
      std::cout << graphName << "," << repArg << ",bfs_parent_s" << s << "_t"
                << t << "," << bt.parent[t] << "\n";
      std::cout << graphName << "," << repArg << ",dfs_parent_s" << s << "_t"
                << t << "," << dt.parent[t] << "\n";
    }
  }

  // Q5: distances between specific pairs.
  std::vector<std::pair<int, int>> pairs = {{10, 20}, {10, 30}, {20, 30}};
  for (auto [a, b] : pairs) {
    if (a > g.n || b > g.n)
      continue;
    std::cout << graphName << "," << repArg << ",distance_" << a << "_" << b
              << "," << getDistance(g, a, b) << "\n";
  }

  // Q6: connected components — count, largest, smallest.
  GraphStats stats = computeStats(g, kExactDiameterBudget);
  std::cout << graphName << "," << repArg << ",component_count,"
            << stats.componentsAmount << "\n";
  std::cout << graphName << "," << repArg << ",component_largest,"
            << stats.largestComponentSize << "\n";
  std::cout << graphName << "," << repArg << ",component_smallest,"
            << stats.smallestComponentSize << "\n";

  // Q7: diameter — both already computed by computeStats above; recomputing
  // them here would double the work. exactDiameter is -1 when it was skipped.
  std::cout << graphName << "," << repArg << ",diameter_approx,"
            << stats.approximateDiameter << "\n";

  if (stats.exactDiameter >= 0) {
    std::cout << graphName << "," << repArg << ",diameter_exact,"
              << stats.exactDiameter << "\n";
  } else {
    std::cerr << "skipping exact diameter (n*(n+m) over the "
              << kExactDiameterBudget << " budget)\n";
  }
}

// --- Memory hold: signal ready, block until released, so an external
//     observer can snapshot this process's RSS while it holds the graph. ---
static void holdForMemory() {
  std::cout << "READY\n";
  std::cout.flush();
  std::string dummy;
  std::getline(std::cin, dummy);
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage: benchmark <graphfile> <list|matrix> "
                 "<time|mem|report|all>\n";
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
      runTiming(*g, graphName, repArg);

    } else if (mode == "mem") {
      holdForMemory();

    } else if (mode == "report") {
      runReport(*g, graphName, repArg);

    } else if (mode == "all") {
      // Memory FIRST — snapshot the pristine graph before timing/report
      // allocate their own structures.
      holdForMemory();
      runTiming(*g, graphName, repArg);
      runReport(*g, graphName, repArg);

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
