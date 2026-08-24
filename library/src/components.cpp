#include "graph.hpp"

std::vector<SearchTree> getComponents(const Graph &g) {
  std::vector<bool> visitedVerteces(g.n, false);
  std::vector<SearchTree> components;
  for (int i = 1; i <= g.n; i++) {
    SearchTree component;
    if (!visitedVerteces[i]) {
      visitedVerteces[i] = true;
      component = bfs(g, i);
      components.push_back(component);
      for (int j = 1; j <= g.n; j++) {
        if (component.parent[j] != 0)
          visitedVerteces[j] = true;
      }
    }
  }
  return components;
}
