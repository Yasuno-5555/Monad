#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include "../UnifiedGrid.hpp"

// v2.0 Two-Asset Model Grid System
// State Space: (m, a, z)
// m: Liquid Asset (Fast moving index)
// a: Illiquid Asset (Middle index)
// z: Income Shock (Slowest index)
// Layout: [ z0(a0(m...), a1(m...)), z1(...) ]

namespace Monad {

class MultiDimGrid {
public:
    UnifiedGrid m_grid; // Liquid Asset Grid
    UnifiedGrid a_grid; // Illiquid Asset Grid
    int n_z;            // Income Process Size

    // Dimensions
    int N_m;
    int N_a;
    int N_z;
    int total_size;

    // Strides for indexing
    int stride_a; // N_m
    int stride_z; // N_m * N_a

    MultiDimGrid() : n_z(0), N_m(0), N_a(0), N_z(0), total_size(0) {}

    MultiDimGrid(const UnifiedGrid& m, const UnifiedGrid& a, int z_dim) 
        : m_grid(m), a_grid(a), n_z(z_dim) {
        
        N_m = m.size;
        N_a = a.size;
        N_z = z_dim;
        
        stride_a = N_m;
        stride_z = N_m * N_a;
        total_size = N_z * stride_z;
    }

    // --- Indexing ---
    
    // (im, ia, iz) -> Flat Index
    // idx = iz * (Na*Nm) + ia * Nm + im
    inline int idx(int im, int ia, int iz) const {
        // Optional bound check could be added here
        return iz * stride_z + ia * stride_a + im;
    }

    // Flat Index -> (im, ia, iz)
    inline void get_coords(int flat_idx, int& im, int& ia, int& iz) const {
        iz = flat_idx / stride_z;
        int rem = flat_idx % stride_z;
        ia = rem / stride_a;
        im = rem % stride_a;
    }
    
    // Helper to get m and a values directly from flat index
    inline std::pair<double, double> get_values(int flat_idx) const {
        int im, ia, iz;
        get_coords(flat_idx, im, ia, iz);
        return {m_grid.nodes[im], a_grid.nodes[ia]};
    }
    
    // Helper used for interpolation loop bounds
    // Returns indices of grid block for a given (ia, iz)
    inline std::pair<int, int> get_block_bounds(int ia, int iz) const {
        int start = idx(0, ia, iz);
        int end = start + N_m; // Exclusive
        return {start, end};
    }
};

} // namespace Monad
