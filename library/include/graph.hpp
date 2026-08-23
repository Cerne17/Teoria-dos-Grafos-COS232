#pragma once
#include <string>
#include <vector>

struct Graph {
  int n;                             // # of vertexes
  int m;                             // # of edges
  std::vector<std::vector<int>> adj; // adjacency list, 1-indexed

  explicit Graph(int vertexCount);
  void addEdge(int u, int v);
  int degree(int u) const;
};

Graph readGraph(const std::string &path);

struct GraphStats {
  int n; // # of vertexes
  int m; // # of edges

  int minDegree;
  int maxDegree;
  double avgDegree;
  double medianDegree;
};

GraphStats computeStats(const Graph &g);

struct SearchTree {
  std::vector<int> parent;
  std::vector<int> level;
};

SearchTree bfs(const Graph &g, int source);

SearchTree dfs(const Graph &g, int source);
