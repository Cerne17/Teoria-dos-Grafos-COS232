#include "graph.hpp"

int getDistance(const Graph &g, const int u, const int v) {
  SearchTree tree = bfs(g, u);
  return tree.level[v]; // -1 means unrecheable
}
