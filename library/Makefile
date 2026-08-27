CXX := g++
CXXFLAGS := -std=c++17 -Iinclude -Wall -Wextra -g

OBJS := src/graph.o src/io.o src/search.o src/stats.o src/components.o src/distance.o app/main.o

.PHONY: all clean

all: app/graphs

app/graphs: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

benchmark: app/benchmark.cpp src/graph.cpp src/io.cpp src/search.cpp src/stats.cpp src/components.cpp
	$(CXX) -std=c++17 -Iinclude -O2 $^ -o app/benchmark

%.o: %.cpp include/graph.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) app/graphs
