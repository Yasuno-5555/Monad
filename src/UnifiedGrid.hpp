#pragma once
#include <vector>

struct UnifiedGrid {
    std::vector<double> nodes;
    int size;

    UnifiedGrid(const std::vector<double>& n) : nodes(n), size(static_cast<int>(n.size())) {}
    UnifiedGrid() : size(0) {}
    
    void resize(int n) {
        nodes.resize(n);
        size = n;
    }
};
