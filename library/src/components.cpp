#include "graph.hpp"
#include <algorithm>
#include <queue>
#include <stack>
#include <vector>

std::vector<std::vector<int>> getComponentsByBreadth(const Graph &g) {
  std::vector<bool> visited(g.n + 1, false);
  std::vector<std::vector<int>> components;

  for (int i = 1; i <= g.n; i++) {
    if (!visited[i]) {
      std::vector<int> component;
      std::queue<int> q;

      visited[i] = true;
      component.push_back(i);
      q.push(i);

      while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (auto v : g.neighbors(u)) {
          if (!visited[v]) {
            visited[v] = true;
            component.push_back(v);
            q.push(v);
          }
        }
      }
      components.push_back(component);
    }
  }
  std::sort(components.begin(), components.end(),
            [](const std::vector<int> &a, const std::vector<int> &b) {
              return a.size() > b.size();
            });
  return components;
}

std::vector<std::vector<int>> getComponentsByDepth(const Graph &g) {
  std::vector<bool> visited(g.n + 1, false);
  std::vector<std::vector<int>> components;

  for (int i = 1; i <= g.n; i++) {
    if (!visited[i]) {
      std::vector<int> component;
      std::stack<int> s;

      s.push(i);

      while (!s.empty()) {
        int u = s.top();
        s.pop();
        if (!visited[u]) {
          visited[u] = true;
          component.push_back(u);

          for (auto v : g.neighbors(u)) {
            if (!visited[v])
              s.push(v);
          }
        }
      }
      components.push_back(component);
    }
  }
  std::sort(components.begin(), components.end(),
            [](const std::vector<int> &a, const std::vector<int> &b) {
              return a.size() > b.size();
            });
  return components;
}
