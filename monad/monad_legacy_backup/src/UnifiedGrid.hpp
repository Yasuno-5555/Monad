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

struct StateSpace {
    int n_a;
    int n_z;
    int size; 

    StateSpace(int na, int nz) : n_a(na), n_z(nz), size(na*nz) {}

    // Global Index: z * Na + a
    int idx(int ia, int iz) const { return iz * n_a + ia; }
    
    // Reverse
    std::pair<int, int> get_coords(int k) const {
        return {k % n_a, k / n_a};
    }
};
