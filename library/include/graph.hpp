#pragma once
#include <memory>
#include <string>
#include <vector>

class Graph {
public:
  int n; // # of vertexes
  int m; // # of edges

  explicit Graph(int vertexCount) : n(vertexCount), m(0) {}
  virtual ~Graph() = default;

  virtual void addEdge(int u, int v) = 0;
  virtual int degree(int u) const = 0;
  virtual std::vector<int> neighbors(int u) const = 0;
  virtual void finalize() = 0;
};

class AdjacencyList : public Graph {
public:
  std::vector<std::vector<int>> adj;

  explicit AdjacencyList(int vertexCount);

  void addEdge(int u, int v) override;
  int degree(int u) const override;
  std::vector<int> neighbors(int u) const override;
  void finalize() override;
};

class AdjacencyMatrix : public Graph {
public:
  std::vector<std::vector<char>> matrix;

  explicit AdjacencyMatrix(int vertexCount);

  void addEdge(int u, int v) override;
  int degree(int u) const override;
  std::vector<int> neighbors(int u) const override;
  void finalize() override;
};

enum class Representation { List, Matrix };

std::unique_ptr<Graph> readGraph(const std::string &path, Representation rep);

struct GraphStats {
  int n; // # of vertexes
  int m; // # of edges

  int minDegree;
  int maxDegree;
  double avgDegree;
  double medianDegree;
  int componentsAmount;
  int largestComponentSize;
  int smallestComponentSize;
  int exactDiameter;
  int approximateDiameter;
};

// exactDiameterLimit: skip the O(n*(n+m)) exact diameter when n exceeds it.
// Pass a non-positive value to always compute it. When skipped, exactDiameter
// comes back as -1 — the same "undefined" convention already used by level
// and getDistance.
GraphStats computeStats(const Graph &g, int exactDiameterLimit = 0);

struct SearchTree {
  std::vector<int> parent;
  std::vector<int> level;
};

SearchTree bfs(const Graph &g, int source);

void bfsExploration(const Graph &g, int source);

SearchTree dfs(const Graph &g, int source);

void dfsExploration(const Graph &g, int source);

void writeStats(const std::string &path, const std::string &graphName,
                const GraphStats &stats);
void writeSearchTree(const std::string &path, const std::string &graphName,
                     const SearchTree &tree);
void writeComponents(const std::string &path, const std::string &graphName,
                     const std::vector<std::vector<int>> &components);

std::vector<std::vector<int>> getComponentsByBreadth(const Graph &g);

std::vector<std::vector<int>> getComponentsByDepth(const Graph &g);

int getDistance(const Graph &g, const int u, const int v);

int getExactDiameter(const Graph &g);

int getApproximateDiameter(const Graph &g);

// Overload for callers that already hold the components, so the BFS sweep that
// builds them is not repeated.
int getApproximateDiameter(const Graph &g,
                           const std::vector<std::vector<int>> &components);
