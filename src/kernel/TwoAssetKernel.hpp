#pragma once
#include <vector>
#include <cmath>
#include "../grid/MultiDimGrid.hpp"
#include "../blocks/FiscalBlock.hpp"

namespace Monad {

struct ThreeAssetParam {
    double beta = 0.986;
    double r_m = -0.02; // Liquid return
    double r_a = 0.02;  // Illiquid return 1
    double r_h = 0.03;  // Illiquid return 2 (Housing/Capital)
    double chi = 0.033; // Adjustment cost scale
    double sigma = 2.0; // CRRA curvature
    double m_min = 0.0; // Borrowing limit
    
    // v3.0 Fiscal with CGT
    FiscalBlock::FiscalPolicy fiscal;
    double capital_gains_tax = 0.15;
    
    // Compute Config
    bool use_gpu = false;
};

struct ThreeAssetPolicy {
    // All flat vectors mapped to (m, a, h, z)
    
    // Value Function
    std::vector<double> value;
    
    // Policy Functions
    std::vector<double> c_pol; // Consumption C(m, a, h, z)
    std::vector<double> m_pol; // Liquid m'(m, a, h, z)
    std::vector<double> a_pol; // Illiquid 1 a'(m, a, h, z)
    std::vector<double> h_pol; // Illiquid 2 h'(m, a, h, z)
    std::vector<double> d_a_pol; // Adjustment d_a = a' - (1+r_a)a
    std::vector<double> d_h_pol; // Adjustment d_h = h' - (1+r_h)h
    
    // 0.0 = No Adjustment, 1.0 = Adjust A, 2.0 = Adjust H, 3.0 = Adjust Both?
    // For simplicity, let's just use it to match the logic we'll implement
    std::vector<double> adjust_flag;

    ThreeAssetPolicy() = default;

    ThreeAssetPolicy(int size) {
        resize(size);
    }
    
    void resize(int size) {
        value.resize(size);
        c_pol.resize(size);
        m_pol.resize(size);
        a_pol.resize(size);
        h_pol.resize(size);
        d_a_pol.resize(size);
        d_h_pol.resize(size);
        adjust_flag.resize(size);
    }
};

using TwoAssetParam = ThreeAssetParam; // Backward compatibility alias if needed temporarily
using TwoAssetPolicy = ThreeAssetPolicy;

} // namespace Monad
