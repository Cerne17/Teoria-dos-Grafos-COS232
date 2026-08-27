#include "graph.hpp"
#include <algorithm>
#include <vector>

GraphStats computeStats(const Graph &g) {
  int n = g.n;
  int m = g.m;

  std::vector<std::vector<int>> components = getComponentsByBreadth(g);
  int componentsAmount = static_cast<int>(components.size());
  int largestComponentSize = static_cast<int>(components.front().size());
  int smallestComponentSize = static_cast<int>(components.back().size());

  int exactDiameter = getExactDiameter(g);
  // Reuse the components computed above instead of sweeping the graph again.
  int approximateDiameter = getApproximateDiameter(g, components);

  std::vector<int> degrees;
  degrees.reserve(n);

  for (int i = 1; i <= n; i++) {
    degrees.push_back(g.degree(i));
  }
  std::sort(degrees.begin(), degrees.end());

  int minDegree = degrees.front();
  int maxDegree = degrees.back();
  double avgDegree = 2.0 * m / n;
  double medianDegree;
  if (n % 2 == 1) {
    medianDegree = degrees[n / 2];
  } else {
    medianDegree = (degrees[n / 2 - 1] + degrees[n / 2]) / 2.0;
  }

  return GraphStats{n,
                    m,
                    minDegree,
                    maxDegree,
                    avgDegree,
                    medianDegree,
                    componentsAmount,
                    largestComponentSize,
                    smallestComponentSize,
                    exactDiameter,
                    approximateDiameter};
}
