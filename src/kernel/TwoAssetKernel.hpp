#pragma once
#include <vector>
#include <cmath>
#include "../grid/MultiDimGrid.hpp"
#include "../blocks/FiscalBlock.hpp"

namespace Monad {

struct TwoAssetParam {
    double beta = 0.986;
    double r_m = -0.02; // Liquid return
    double r_a = 0.02; // Illiquid return
    double chi = 0.033; // Adjustment cost scale
    double sigma = 2.0; // CRRA curvature
    double m_min = 0.0; // Borrowing limit
    
    // v2.1 Fiscal
    FiscalBlock::FiscalPolicy fiscal;
    
    // v3.0 Compute Config
    bool use_gpu = false;
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

    TwoAssetPolicy() = default;

    TwoAssetPolicy(int size) {
        resize(size);
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
