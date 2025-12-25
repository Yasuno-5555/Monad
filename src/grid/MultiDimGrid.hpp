#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include "../UnifiedGrid.hpp"

// v3.0 Three-Asset Model Grid System
// State Space: (m, a, h, z)
// m: Liquid Asset (Fastest moving)
// a: Illiquid Asset 1
// h: Illiquid Asset 2 (Housing/Capital)
// z: Income Shock (Slowest index)
// Layout: [ z(h(a(m...))) ]

namespace Monad {

class MultiDimGrid {
public:
    UnifiedGrid m_grid; 
    UnifiedGrid a_grid; 
    UnifiedGrid h_grid; // Asset 3
    int n_z;            

    // Dimensions
    int N_m, N_a, N_h, N_z;
    int total_size;

    // Strides
    int stride_a; // N_m
    int stride_h; // N_m * N_a
    int stride_z; // N_m * N_a * N_h

    MultiDimGrid() : n_z(0), N_m(0), N_a(0), N_h(0), N_z(0), total_size(0) {}

    MultiDimGrid(const UnifiedGrid& m, const UnifiedGrid& a, const UnifiedGrid& h, int z_dim) 
        : m_grid(m), a_grid(a), h_grid(h), n_z(z_dim) {
        
        N_m = m.size;
        N_a = a.size;
        N_h = h.size;
        N_z = z_dim;
        
        stride_a = N_m;
        stride_h = N_m * N_a;
        stride_z = N_m * N_a * N_h;
        total_size = N_z * stride_z;
    }

    // --- Indexing ---
    
    // (im, ia, ih, iz) -> Flat Index
    inline int idx(int im, int ia, int ih, int iz) const {
        return iz * stride_z + ih * stride_h + ia * stride_a + im;
    }

    // Flat Index -> (im, ia, ih, iz)
    inline void get_coords(int flat_idx, int& im, int& ia, int& ih, int& iz) const {
        iz = flat_idx / stride_z;
        int rem = flat_idx % stride_z;
        ih = rem / stride_h;
        rem %= stride_h;
        ia = rem / stride_a;
        im = rem % stride_a;
    }
    
    inline std::tuple<double, double, double> get_values(int flat_idx) const {
        int im, ia, ih, iz;
        get_coords(flat_idx, im, ia, ih, iz);
        return {m_grid.nodes[im], a_grid.nodes[ia], h_grid.nodes[ih]};
    }
    
    // Returns indices of grid block for a given (ia, ih, iz)
    inline std::pair<int, int> get_block_bounds(int ia, int ih, int iz) const {
        int start = idx(0, ia, ih, iz);
        int end = start + N_m; 
        return {start, end};
    }
};

} // namespace Monad
