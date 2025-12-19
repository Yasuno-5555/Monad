#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include "../UnifiedGrid.hpp"

// Phase 5: Lightweight 3-Asset HANK Grid (5D)
// State Space: (m, as, ac, z, b)
// m:  Liquid Asset (Fastest moving index, Coalesced)
// as: Stock Asset
// ac: Crypto Asset
// z:  Income Shock
// b:  Belief Bias (Slowest index)
// Layout: [ b(z(ac(as(m...)))) ]

namespace Monad {

class MultiDimGrid5D {
public:
    UnifiedGrid m_grid;  // Liquid
    UnifiedGrid as_grid; // Stock
    UnifiedGrid ac_grid; // Crypto
    UnifiedGrid b_grid;  // Belief Bias
    int n_z;             // Income

    // Dimensions
    int N_m;
    int N_as;
    int N_ac;
    int N_z;
    int N_b;
    int total_size;

    // Strides (Coalesced Optimized: m is innermost)
    int stride_as; // N_m
    int stride_ac; // N_m * N_as
    int stride_z;  // N_m * N_as * N_ac
    int stride_b;  // N_m * N_as * N_ac * N_z

    MultiDimGrid5D() : N_m(0), N_as(0), N_ac(0), N_z(0), N_b(0), total_size(0) {}

    MultiDimGrid5D(const UnifiedGrid& m, const UnifiedGrid& as, const UnifiedGrid& ac, int z_dim, const UnifiedGrid& b) 
        : m_grid(m), as_grid(as), ac_grid(ac), n_z(z_dim), b_grid(b) {
        
        N_m  = m.size;
        N_as = as.size;
        N_ac = ac.size;
        N_z  = z_dim;
        N_b  = b.size;
        
        // Layout: (b, z, ac, as, m)
        stride_as = N_m;
        stride_ac = N_m * N_as;
        stride_z  = N_m * N_as * N_ac;
        stride_b  = N_m * N_as * N_ac * N_z;
        total_size = N_b * stride_b;
    }

    // --- Indexing ---
    
    // (im, ias, iac, iz, ib) -> Flat Index
    inline int idx(int im, int ias, int iac, int iz, int ib) const {
        return ib * stride_b + iz * stride_z + iac * stride_ac + ias * stride_as + im;
    }

    // Flat Index -> Coords
    inline void get_coords(int flat_idx, int& im, int& ias, int& iac, int& iz, int& ib) const {
        ib = flat_idx / stride_b;
        int rem_b = flat_idx % stride_b;
        
        iz = rem_b / stride_z;
        int rem_z = rem_b % stride_z;
        
        iac = rem_z / stride_ac;
        int rem_c = rem_z % stride_ac;
        
        ias = rem_c / stride_as;
        im  = rem_c % stride_as;
    }
    
    // Helper to get values directly
    inline std::tuple<double, double, double, double> get_values(int flat_idx) const {
        int im, ias, iac, iz, ib;
        get_coords(flat_idx, im, ias, iac, iz, ib);
        // Note: z is index only here, z_val usually from income process
        return std::make_tuple(m_grid.nodes[im], as_grid.nodes[ias], ac_grid.nodes[iac], b_grid.nodes[ib]);
    }
};

} // namespace Monad
