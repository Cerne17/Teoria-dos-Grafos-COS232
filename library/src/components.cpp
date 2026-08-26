#include "graph.hpp"
#include <queue>
#include <stack>

std::vector<bool> auxiliarBFS(const Graph &g, const int source) {
  /* Used to retrieve all visited vertexes */
  std::queue<int> q;
  std::vector<bool> visited(g.n + 1, false);

  q.push(source);
  visited[source] = true;

  while (!q.empty()) {
    int current = q.front();
    q.pop();
    visited[current] = true;

    for (auto v : g.neighbors(current)) {
      if (!visited[v]) {
        q.push(v);
        visited[v] = true;
      }
    }
  }
  return visited;
}

std::vector<std::vector<int>> getComponentsByBreadth(const Graph &g) {
  std::vector<bool> visited(g.n + 1, false);
  std::vector<std::vector<int>> components;
  for (int i = 1; i <= g.n; i++) {
    if (!visited[i]) {
      std::vector<bool> reached = auxiliarBFS(g, i);
      std::vector<int> currentComponent;
      for (int j = 1; j <= g.n; j++) {
        if (reached[j]) {
          visited[j] = true;
          currentComponent.push_back(j);
        }
      }
      components.push_back(currentComponent);
    }
  }
  return components;
}

std::vector<bool> auxiliarDFS(const Graph &g, const int source) {
  /* auxiliar DFS to retrieve all visited vertexes */
  std::stack<int> s;
  std::vector<bool> visited(g.n + 1, false);

  s.push(source);

  while (!s.empty()) {
    int current = s.top();
    s.pop();
    visited[current] = true;

    for (auto u : g.neighbors(current)) {
      if (!visited[u]) {
        s.push(u);
      }
    }
  }

  return visited;
}

std::vector<std::vector<int>> getComponentsByDepth(const Graph &g) {
  std::vector<bool> visited(g.n + 1, false);
  std::vector<std::vector<int>> components;

  for (int i = 1; i <= g.n; i++) {
    std::vector<int> currentComponent;
    std::vector<bool> reached = auxiliarDFS(g, i);
    if (!visited[i]) {

      for (int j = 1; j <= g.n; j++)
        if (reached[j]) {
          currentComponent.push_back(j);
          visited[j] = true;
        }
      components.push_back(currentComponent);
    }
  }
  return components;
}
