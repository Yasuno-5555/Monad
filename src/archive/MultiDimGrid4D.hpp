#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include "../UnifiedGrid.hpp"

// Phase 4: Belief-Distribution HANK Grid (4D)
// State Space: (m, a, z, b)
// m: Liquid Asset (Fastest moving index) [Thread coalescing opt]
// a: Illiquid Asset
// z: Income Shock
// b: Belief Bias (Slowest index)
// Layout: [ b0(z0(a0(m...))), b1(...) ]

namespace Monad {

class MultiDimGrid4D {
public:
    UnifiedGrid m_grid; // Liquid Asset Grid
    UnifiedGrid a_grid; // Illiquid Asset Grid
    UnifiedGrid b_grid; // Belief Grid (NEW)
    int n_z;            // Income Process Size

    // Dimensions
    int N_m;
    int N_a;
    int N_z;
    int N_b; // NEW
    int total_size;

    // Strides for indexing (Memory Coalescing Optimized: m is innermost)
    int stride_a; // N_m
    int stride_z; // N_m * N_a
    int stride_b; // N_m * N_a * N_z

    MultiDimGrid4D() : n_z(0), N_m(0), N_a(0), N_z(0), N_b(0), total_size(0) {}

    MultiDimGrid4D(const UnifiedGrid& m, const UnifiedGrid& a, int z_dim, const UnifiedGrid& b) 
        : m_grid(m), a_grid(a), n_z(z_dim), b_grid(b) {
        
        N_m = m.size;
        N_a = a.size;
        N_z = z_dim;
        N_b = b.size;
        
        // Coalesced Layout: (b, z, a, m)
        // b: Slowest
        // m: Fastest
        stride_a = N_m;
        stride_z = N_m * N_a;
        stride_b = N_m * N_a * N_z;
        total_size = N_b * stride_b;
    }

    // --- Indexing ---
    
    // (im, ia, iz, ib) -> Flat Index
    inline int idx(int im, int ia, int iz, int ib) const {
        return ib * stride_b + iz * stride_z + ia * stride_a + im;
    }

    // Flat Index -> (im, ia, iz, ib)
    inline void get_coords(int flat_idx, int& im, int& ia, int& iz, int& ib) const {
        ib = flat_idx / stride_b;
        int rem_b = flat_idx % stride_b;
        
        iz = rem_b / stride_z;
        int rem_z = rem_b % stride_z;
        
        ia = rem_z / stride_a;
        im = rem_z % stride_a;
    }
    
    // Helper to get m, a, b values directly from flat index
    inline std::tuple<double, double, double> get_values(int flat_idx) const {
        int im, ia, iz, ib;
        get_coords(flat_idx, im, ia, iz, ib);
        return std::make_tuple(m_grid.nodes[im], a_grid.nodes[ia], b_grid.nodes[ib]);
    }
    
    // Returns indices of grid block for a given (ia, iz, ib)
    inline std::pair<int, int> get_block_bounds(int ia, int iz, int ib) const {
        int start = idx(0, ia, iz, ib);
        int end = start + N_m; // Exclusive
        return {start, end};
    }
};

} // namespace Monad
