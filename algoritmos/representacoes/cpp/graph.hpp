#pragma once
#include <vector>

struct Graph {
  int n; // # of vertexes
  int m; // # of edges
  std::vector<std::vector<int>> adj; // adjacency list, 1-indexed

  explicit Graph(int vertexCount);
  void addEdge(int u, int v);
  int degree(int u) const;
};
