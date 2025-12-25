#pragma once
#include <vector>
#include "backend/TwoAssetBackend.hpp"
#include "grid/MultiDimGrid.hpp"
#include "Params.hpp"

namespace Monad {

class CpuBackend : public ThreeAssetBackend {
    const MultiDimGrid& grid;
    const ThreeAssetParam& params;
    
    // Internal buffers
    std::vector<double> E_V_next;
    std::vector<double> E_Vm_next;

public:
    CpuBackend(const MultiDimGrid& g, const ThreeAssetParam& p);

    double solve_bellman_iteration(const ThreeAssetPolicy& guess, 
                                   ThreeAssetPolicy& result, 
                                   const IncomeProcess& income) override;

private:
    void compute_expectations(const ThreeAssetPolicy& next_pol, const IncomeProcess& income);
    void solve_no_adjust_slice(int iz, int ia, int ih, double z_val, ThreeAssetPolicy& res);
    void solve_adjust_slice(int iz, int ia, int ih, double z_val, ThreeAssetPolicy& res);
    void finalize_policy(const ThreeAssetPolicy& old, ThreeAssetPolicy& res, double& diff);

    // Helpers
    double u(double c) const;
    double inv_u_prime(double val) const;
    double adj_cost(double d, double a_curr) const;

    double interp_1d(const std::vector<double>& x, const std::vector<double>& y, double xi) const;
    double interp_1d_slice_m(const std::vector<double>& data, int iz, int ia, int ih, double m) const;
    double interpolate_2d_m_a(const std::vector<double>& data, int iz, int ih, double m, double a) const;
    double interpolate_3d_m_a_h(const std::vector<double>& data, int iz, double m, double a, double h) const;
};

} // namespace Monad
