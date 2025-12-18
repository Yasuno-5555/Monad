#pragma once
#include <vector>
#include <cmath>
#include "../grid/MultiDimGrid.hpp"
#include "../blocks/FiscalBlock.hpp"

// v2.0 Two-Asset Kernel
// Data structures for policy and value functions in the two-asset world.

namespace Monad {

struct TwoAssetParam {
    double beta;
    double r_m; // Liquid return
    double r_a; // Illiquid return
    double chi; // Adjustment cost scale
    double sigma; // CRRA curvature
    double m_min; // Borrowing limit
    
    // v2.1 Fiscal
    FiscalBlock::FiscalPolicy fiscal;
};

struct TwoAssetPolicy {
    // All flat vectors mapped to (m, a, z)
    
    // Value Function
    std::vector<double> value;
    
    // Policy Functions
    std::vector<double> c_pol; // Consumption C(m, a, z)
    std::vector<double> m_pol; // Liquid Savings m'(m, a, z)
    std::vector<double> a_pol; // Illiquid Savings a'(m, a, z)
    std::vector<double> d_pol; // Adjustment d = a' - (1+r_a)a
    
    // Region Indicator
    // 0.0 = No Adjustment (Inaction)
    // 1.0 = Adjustment
    std::vector<double> adjust_flag;

    TwoAssetPolicy(int size) {
        value.resize(size);
        c_pol.resize(size);
        m_pol.resize(size);
        a_pol.resize(size);
        d_pol.resize(size);
        adjust_flag.resize(size);
    }
    
    void resize(int size) {
        value.resize(size);
        c_pol.resize(size);
        m_pol.resize(size);
        a_pol.resize(size);
        d_pol.resize(size);
        adjust_flag.resize(size);
    }
};

} // namespace Monad
