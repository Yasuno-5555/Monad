#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "AnalyticalSolver.hpp"
#include "grid/MultiDimGrid.hpp"
#include "kernel/TwoAssetKernel.hpp"
#include "solver/TwoAssetSolver.hpp"
#include "solver/OneAssetSolver.hpp"
#include "aggregator/DistributionAggregator3D.hpp"
#include "ssj/SsjSolver3D.hpp"
#include "ssj/GeneralEquilibrium.hpp"
#include "ssj/SparseMatrixBuilder.hpp"
#include "InequalityAnalyzer.hpp"
#include "MicroAnalyzer.hpp"
#include "FiscalExperiment.hpp"
#include "OptimalPolicy.hpp"
#include "Params.hpp"
#include "solver/GenericNewtonSolver.hpp"

namespace py = pybind11;

// ============================================================================
// Helpers
// ============================================================================

UnifiedGrid make_power_grid(int size, double min, double max, double curv) {
    UnifiedGrid g;
    g.resize(size);
    if (size == 1) {
        g.nodes[0] = min;
        return g;
    }
    for(int i = 0; i < size; ++i) {
        double t = (double)i / (size - 1);
        g.nodes[i] = min + (max - min) * std::pow(t, curv);
    }
    return g;
}

Monad::MultiDimGrid reconstruct_grid(
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    int Nz) 
{
    auto m_grid = make_power_grid(Nm, m_min, m_max, m_curv);
    auto a_grid = make_power_grid(Na, a_min, a_max, a_curv);
    return Monad::MultiDimGrid(m_grid, a_grid, Nz);
}

IncomeProcess reconstruct_income(const std::vector<double>& z_grid, const std::vector<double>& Pi_flat) {
    IncomeProcess income;
    income.n_z = z_grid.size();
    income.z_grid = z_grid;
    income.Pi_flat = Pi_flat;
    return income;
}

Monad::TwoAssetParam reconstruct_params(
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer, double m_min) 
{
    Monad::TwoAssetParam params;
    params.beta = beta;
    params.r_m = r_m;
    params.r_a = r_a;
    params.chi = chi;
    params.sigma = sigma;
    params.m_min = m_min;
    params.fiscal.tax_rule.lambda = tax_lambda;
    params.fiscal.tax_rule.tau = tax_tau;
    params.fiscal.tax_rule.transfer = tax_transfer;
    return params;
}

Monad::TwoAssetPolicy reconstruct_policy(
    const Monad::MultiDimGrid& grid,
    const std::vector<double>& c_pol,
    const std::vector<double>& m_pol,
    const std::vector<double>& a_pol,
    const std::vector<double>& value,
    const std::vector<double>& adjust_flag)
{
    Monad::TwoAssetPolicy policy(grid.total_size);
    policy.c_pol = c_pol;
    policy.m_pol = m_pol;
    policy.a_pol = a_pol;
    policy.value = value;
    policy.adjust_flag = adjust_flag;
    return policy;
}

// ============================================================================
// 1. Household Block Wrappers
// ============================================================================

py::dict solve_hank_steady_state_py(
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    int max_bellman_iter, double bellman_tol,
    int max_dist_iter, double dist_tol
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    auto params = reconstruct_params(beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, m_min);

    Monad::TwoAssetPolicy policy(grid.total_size);
    for(int i = 0; i < grid.total_size; ++i) policy.c_pol[i] = 0.1;
    Monad::TwoAssetPolicy next_policy(grid.total_size);

    Monad::TwoAssetSolver solver(grid, params, nullptr);
    for(int iter = 0; iter < max_bellman_iter; ++iter) {
        double diff = solver.solve_bellman(policy, next_policy, income);
        policy = next_policy;
        if(diff < bellman_tol) break;
    }

    Monad::DistributionAggregator3D aggregator(grid);
    std::vector<double> D = aggregator.init_uniform();
    std::vector<double> D_next(grid.total_size);
    for(int iter = 0; iter < max_dist_iter; ++iter) {
        double diff = aggregator.forward_iterate(D, D_next, policy, income);
        D = D_next;
        if(diff < dist_tol) break;
    }

    double Agg_M, Agg_A;
    aggregator.compute_aggregates(D, Agg_M, Agg_A);
    
    // Gini Calculation (Approximate)
    // Needs sorted wealth and PDF. Implemented simply here using DistributionUtils logic if available.
    // For now, let Python handle Gini from D.

    py::dict result;
    result["c_pol"] = policy.c_pol;
    result["m_pol"] = policy.m_pol;
    result["a_pol"] = policy.a_pol;
    result["value"] = policy.value;
    result["adjust_flag"] = policy.adjust_flag;
    result["distribution"] = D;
    result["agg_liquid"] = Agg_M;
    result["agg_illiquid"] = Agg_A;
    result["E_Vm"] = solver.E_Vm_next;
    result["E_V"] = solver.E_V_next;
    
    return result;
}



// 1.b One-Asset HANK Wrapper
py::dict solve_one_asset_steady_state_py(
    int Nm, double m_min, double m_max, double m_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    int max_bellman_iter, double bellman_tol,
    int max_dist_iter, double dist_tol
) {
    // Force Na=1, a_min=0, a_max=0
    int Na = 1; 
    double a_min = 0.0, a_max = 0.0, a_curv = 1.0;
    
    // a_grid will have 1 point at 0.0
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    
    // Params: pass dummy r_a, chi
    auto params = reconstruct_params(beta, r_m, 0.0, 0.0, sigma, tax_lambda, tax_tau, tax_transfer, m_min);

    Monad::TwoAssetPolicy policy(grid.total_size);
    for(int i = 0; i < grid.total_size; ++i) policy.c_pol[i] = 0.1;
    Monad::TwoAssetPolicy next_policy(grid.total_size);

    Monad::OneAssetSolver solver(grid, params);
    for(int iter = 0; iter < max_bellman_iter; ++iter) {
        double diff = solver.solve_bellman(policy, next_policy, income);
        policy = next_policy;
        if(diff < bellman_tol) break;
    }

    Monad::DistributionAggregator3D aggregator(grid);
    std::vector<double> D = aggregator.init_uniform();
    std::vector<double> D_next(grid.total_size);
    for(int iter = 0; iter < max_dist_iter; ++iter) {
        double diff = aggregator.forward_iterate(D, D_next, policy, income);
        D = D_next;
        if(diff < dist_tol) break;
    }

    double Agg_M, Agg_A;
    aggregator.compute_aggregates(D, Agg_M, Agg_A);

    py::dict result;
    result["c_pol"] = policy.c_pol;
    result["m_pol"] = policy.m_pol;
    result["a_pol"] = policy.a_pol;
    result["value"] = policy.value;
    result["distribution"] = D;
    result["agg_liquid"] = Agg_M;
    result["E_Vm"] = solver.E_Vm_next;
    result["E_V"] = solver.E_V_next;
    
    return result;
}

// 1.c Probe Policy at arbitrary state
double probe_policy_py(
    const std::vector<double>& pol_data, // Flat policy vector
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    int Nz,
    double m_val, double a_val, int z_idx
) {
    // Reconstruct minimal components for interpolation
    // This is expensive to do every call, but okay for interactive probes
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, Nz);
    
    // Use Solver's static helper or reuse interpolation logic
    // But interpolation is private in Solver. 
    // We'll reimplement a simple 2D interpolation here for the API.
    
    if (z_idx < 0 || z_idx >= Nz) return 0.0;
    
    // Reuse Monad::TwoAssetSolver logic? It is not static.
    // Let's instantiate a solver with minimal params just to use helpers? No, overhead.
    // Hardcode 2D interp here.
    
    // Helper: 1D
    auto interp_1d = [](const std::vector<double>& nodes, double val) -> std::pair<int, double> {
         if(val <= nodes.front()) return {0, 0.0};
         if(val >= nodes.back()) return {(int)nodes.size()-1, 0.0};
         auto it = std::lower_bound(nodes.begin(), nodes.end(), val);
         int i = std::distance(nodes.begin(), it);
         double t = (val - nodes[i-1]) / (nodes[i] - nodes[i-1]);
         return {i, t};
    };
    
    auto p_m = interp_1d(grid.m_grid.nodes, m_val);
    auto p_a = interp_1d(grid.a_grid.nodes, a_val);
    
    // Bilinear
    // (im-1, ia-1), (im, ia-1), (im-1, ia), (im, ia)
    // Logic is slightly complex due to index mapping.
    // Just simple nearest neighbor for robustness? No, use linear.
    
    int im = (p_m.second == 0.0 && p_m.first == 0) ? 0 : p_m.first; // upper index
    int im_prev = (im == 0) ? 0 : im - 1;
    double tm = p_m.second;
    
    int ia = (p_a.second == 0.0 && p_a.first == 0) ? 0 : p_a.first;
    int ia_prev = (ia == 0) ? 0 : ia - 1;
    double ta = p_a.second;
    
    int idx00 = grid.idx(im_prev, ia_prev, z_idx);
    int idx10 = grid.idx(im,     ia_prev, z_idx);
    int idx01 = grid.idx(im_prev, ia,     z_idx);
    int idx11 = grid.idx(im,     ia,     z_idx);
    
    double v00 = pol_data[idx00];
    double v10 = pol_data[idx10];
    double v01 = pol_data[idx01];
    double v11 = pol_data[idx11];
    
    double v0 = v00 + tm * (v10 - v00); // Interp along m at ia_prev
    double v1 = v01 + tm * (v11 - v01); // Interp along m at ia
    
    return v0 + ta * (v1 - v0);
}


// ============================================================================
// 2. SSJ Block Wrappers
// ============================================================================

// 2.a Transition Matrix Export
py::tuple get_transition_matrix_py(
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma, double m_min_param,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& m_pol, const std::vector<double>& a_pol
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    
    Monad::TwoAssetPolicy policy(grid.total_size);
    policy.m_pol = m_pol;
    policy.a_pol = a_pol;
    
    Monad::SparseMatrixBuilder builder(grid, income);
    Monad::SparseMat Lambda = builder.build_transition_matrix(policy);
    
    std::vector<int> rows, cols;
    std::vector<double> data;
    
    // Iterate sparse matrix
    for (int k=0; k<Lambda.outerSize(); ++k) {
        for (Monad::SparseMat::InnerIterator it(Lambda, k); it; ++it) {
            rows.push_back(it.row());
            cols.push_back(it.col());
            data.push_back(it.value());
        }
    }
    
    return py::make_tuple(rows, cols, data, grid.total_size);
}

// 2.b Compute Jacobians (Existing but refactored for clarity)
py::dict compute_jacobians_py(
    // ... same args as before ...
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& c_pol, const std::vector<double>& m_pol, const std::vector<double>& a_pol,
    const std::vector<double>& value, const std::vector<double>& adjust_flag,
    const std::vector<double>& distribution,
    const std::vector<double>& E_Vm, const std::vector<double>& E_V,
    int T
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    auto params = reconstruct_params(beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, m_min);
    auto policy = reconstruct_policy(grid, c_pol, m_pol, a_pol, value, adjust_flag);

    Monad::SsjSolver3D ssj_solver(grid, params, income, policy, distribution, E_Vm, E_V);
    auto J = ssj_solver.compute_block_jacobians(T);

    py::dict result;
    if(J.count("C") && J["C"].count("rm")) result["J_C_rm"] = J["C"]["rm"];
    if(J.count("C") && J["C"].count("w"))  result["J_C_w"]  = J["C"]["w"];
    if(J.count("B") && J["B"].count("rm")) result["J_B_rm"] = J["B"]["rm"];
    
    return result;
}


// ============================================================================
// 3. GE Solver Wrappers
// ============================================================================

// 3.a Solve with ZLB
py::dict solve_ge_zlb_py(
    Eigen::MatrixXd J_C_rm, Eigen::MatrixXd J_C_w, // Raw Jacobians
    Eigen::VectorXd dr_star,       // Natural rate shock
    double beta, double kappa, double phi_pi,
    std::vector<int> forced_binding // Optional FG
) {
    // Reconstruct SsjSolver3D? 
    // Wait, GeneralEquilibrium needs reference to SsjSolver3D.
    // The previous implementation of solve_with_zlb in GeneralEquilibrium.hpp
    // REQUIRED `solver.compute_block_jacobians(T)`.
    // It recomputes them inside. Ideally we pass precomputed Js.
    // BUT the GE class logic uses `solver` reference.
    
    // We can refactor `GeneralEquilibrium` to accept J matrices directly,
    // OR we just assume standard NK block logic here in the wrapper (easier).
    
    // Let's implement the ZLB logic directly here using the J matrices provided.
    // This duplicates logic in `GeneralEquilibrium.hpp` but decouples dependencies.
    // Actually, `GeneralEquilibrium.hpp` has the logic. Let's adapt it to use passed matrices.
    
    int T = dr_star.size();
    
    // NKPC Matrices
    Eigen::MatrixXd M_pi_Y = Monad::SimpleBlocks::build_nkpc_jacobian(T, beta, kappa);
    
    // Fisher & Taylor
    Eigen::MatrixXd L_inv = Eigen::MatrixXd::Zero(T, T);
    for(int t=0; t<T-1; ++t) L_inv(t, t+1) = 1.0;
    
    Eigen::MatrixXd Term_r_Y = (phi_pi * Eigen::MatrixXd::Identity(T,T) - L_inv) * M_pi_Y;
    Eigen::MatrixXd A_GE = Eigen::MatrixXd::Identity(T, T) - J_C_w - J_C_rm * Term_r_Y;
    Eigen::MatrixXd J_Y_eps = A_GE.colPivHouseholderQr().solve(J_C_rm);
    Eigen::MatrixXd J_Y_rstar = J_Y_eps; 
    
    Eigen::MatrixXd J_istar_eps = phi_pi * M_pi_Y * J_Y_eps;
    Eigen::VectorXd istar_base = phi_pi * M_pi_Y * (J_Y_rstar * dr_star) + dr_star;
    
    // ZLB Loop (Same as GeneralEquilibrium.hpp)
    Eigen::VectorXd eps = Eigen::VectorXd::Zero(T);
    std::vector<int> binding_idx = forced_binding;
    
    for(int iter=0; iter<20; ++iter) {
        Eigen::VectorXd istar = istar_base + J_istar_eps * eps;
        std::vector<int> next_binding = forced_binding;
        
        for(int t=0; t<T; ++t) {
            bool is_forced = false;
            for(int fb : forced_binding) if(fb == t) is_forced = true;
            if(is_forced) continue;

            double i_val = istar[t] + eps[t]; 
            if (i_val < -1e-6) next_binding.push_back(t);
            else if (eps[t] > 1e-6) next_binding.push_back(t); 
        }
        
        std::sort(next_binding.begin(), next_binding.end());
        next_binding.erase(std::unique(next_binding.begin(), next_binding.end()), next_binding.end());
        
        if (next_binding == binding_idx && iter > 0) break;
        binding_idx = next_binding;
        
        eps = Eigen::VectorXd::Zero(T); 
        int n_bind = binding_idx.size();
        
        if (n_bind > 0) {
            Eigen::MatrixXd H_sub(n_bind, n_bind);
            Eigen::VectorXd rhs_sub(n_bind);
            Eigen::MatrixXd H = J_istar_eps + Eigen::MatrixXd::Identity(T, T);
            
            for(int r=0; r<n_bind; ++r) {
                int tr = binding_idx[r];
                rhs_sub(r) = -istar_base(tr);
                for(int c=0; c<n_bind; ++c) H_sub(r, c) = H(tr, binding_idx[c]);
            }
            Eigen::VectorXd eps_sub = H_sub.colPivHouseholderQr().solve(rhs_sub);
            
            for(int r=0; r<n_bind; ++r) {
                int t_idx = binding_idx[r];
                bool is_forced = false;
                for(int fb : forced_binding) if(fb == t_idx) is_forced = true;
                if (eps_sub(r) > 0.0 || is_forced) eps(t_idx) = eps_sub(r);
            }
        }
    }
    
    Eigen::VectorXd Y = J_Y_rstar * dr_star + J_Y_eps * eps;
    Eigen::VectorXd i_nom = istar_base + J_istar_eps * eps + eps; 
    
    py::dict result;
    result["dY"] = Y;
    result["i"] = i_nom;
    result["eps"] = eps;
    return result;
}


// ============================================================================
// 4. Inequality Analysis Wrappers
// ============================================================================

py::dict analyze_inequality_py(
    // ... Standard Grid/Param/State args ...
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& c_pol, const std::vector<double>& m_pol, const std::vector<double>& a_pol,
    const std::vector<double>& value, const std::vector<double>& adjust_flag,
    const std::vector<double>& distribution,
    const std::vector<double>& E_Vm, const std::vector<double>& E_V,
    // Paths
    const std::vector<double>& dr_path, // Real Rate Path
    const std::vector<double>& dZ_path  // Income/Wage Path
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    auto params = reconstruct_params(beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, m_min);
    auto policy = reconstruct_policy(grid, c_pol, m_pol, a_pol, value, adjust_flag);

    // reconstruct solver to get jac_builder partials
    Monad::SsjSolver3D ssj_solver(grid, params, income, policy, distribution, E_Vm, E_V);
    
    // This is expensive: we recompte partials just for analysis. 
    // Ideally we cache partials or pass them.
    // But for "Analysis" step, speed is less critical than Solver loop.
    
    // We need to access JBuilder from SsjSolver. It is private.
    // Workaround: In SsjSolver structure (which I edited earlier), jac_builder is private.
    // I should probably have made it public or add an accessor.
    // BUT, SsjSolver3D::compute_block_jacobians uses it.
    // InequalityAnalyzer needs `PartialMap`.
    
    // Let's modify InequalityAnalyzer usage pattern.
    // Or we can just compute partials locally here if we instantiate JacobianBuilder separately.
    
    Monad::JacobianBuilder3D builder(grid, params, income, E_Vm, E_V);
    auto partials = builder.compute_partials(policy);
    
    Monad::InequalityAnalyzer analyzer(grid, distribution, policy, partials);
    
    // Prepare paths map
    std::map<std::string, Eigen::VectorXd> paths;
    int T = dr_path.size();
    paths["m"] = Eigen::Map<const Eigen::VectorXd>(dr_path.data(), T); // map 'm' (rate) to 'm' input
    paths["rm"] = paths["m"]; // alias
    
    // Wage/Income path. This affects 'w' or 'z'?
    // In JacobianBuilder, usually 'w' inputs affect budget.
    // Assuming dZ_path is mapped to 'w' (wage)
    paths["w"] = Eigen::Map<const Eigen::VectorXd>(dZ_path.data(), T);
    
    auto groups = analyzer.analyze_consumption_response(paths);
    
    // Also Heatmap at peak (argmax dZ)
    int t_peak = 0;
    double max_z = -1e9;
    for(int t=0; t<T; ++t) if(std::abs(dZ_path[t]) > max_z) { max_z = std::abs(dZ_path[t]); t_peak = t; }
    
    auto heatmap = analyzer.compute_consumption_heatmap(paths, t_peak);
    
    py::dict result;
    result["top10"]    = groups.top10;
    result["bottom50"] = groups.bottom50;
    result["debtors"]  = groups.debtors;
    result["heatmap"]  = heatmap;
    result["t_peak"]   = t_peak;
    
    return result;
}


// ============================================================================
// 5. Advanced Experiments Wrappers
// ============================================================================

// 5.a Fiscal Shock
py::dict solve_fiscal_shock_py(
    // Grid/Param boilerplate (reused)
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& c_pol, const std::vector<double>& m_pol, const std::vector<double>& a_pol,
    const std::vector<double>& value, const std::vector<double>& adjust_flag,
    const std::vector<double>& distribution,
    const std::vector<double>& E_Vm, const std::vector<double>& E_V,
    // Paths
    const std::vector<double>& dG_path,
    const std::vector<double>& dTrans_path
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    auto params = reconstruct_params(beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, m_min);
    auto policy = reconstruct_policy(grid, c_pol, m_pol, a_pol, value, adjust_flag);

    Monad::SsjSolver3D ssj_solver(grid, params, income, policy, distribution, E_Vm, E_V);
    
    int T = dG_path.size();
    Monad::FiscalExperiment experiment(ssj_solver, T);
    
    Eigen::VectorXd G_vec = Eigen::Map<const Eigen::VectorXd>(dG_path.data(), T);
    Eigen::VectorXd T_vec = Eigen::Map<const Eigen::VectorXd>(dTrans_path.data(), T);
    
    auto res = experiment.solve_fiscal_shock(G_vec, T_vec);
    
    py::dict result;
    result["dY"] = res.dY;
    result["dC"] = res.dC;
    result["multiplier"] = res.multiplier;
    return result;
}

// 5.b Multiplier Decomposition
py::dict decompose_multiplier_py(
    // Grid/Param boilerplate
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& c_pol, const std::vector<double>& m_pol, const std::vector<double>& a_pol,
    const std::vector<double>& value, const std::vector<double>& adjust_flag,
    const std::vector<double>& distribution,
    const std::vector<double>& E_Vm, const std::vector<double>& E_V,
    // Inputs
    const std::vector<double>& dY_path,
    const std::vector<double>& dTrans_path,
    const std::vector<double>& dr_path
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    auto params = reconstruct_params(beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, m_min);
    auto policy = reconstruct_policy(grid, c_pol, m_pol, a_pol, value, adjust_flag);

    Monad::SsjSolver3D ssj_solver(grid, params, income, policy, distribution, E_Vm, E_V);
    
    int T = dY_path.size();
    Monad::FiscalExperiment experiment(ssj_solver, T);
    
    Eigen::VectorXd Y_vec = Eigen::Map<const Eigen::VectorXd>(dY_path.data(), T);
    Eigen::VectorXd Tr_vec = Eigen::Map<const Eigen::VectorXd>(dTrans_path.data(), T);
    Eigen::VectorXd r_vec = Eigen::Map<const Eigen::VectorXd>(dr_path.data(), T);
    
    auto res = experiment.decompose_multiplier(Y_vec, Tr_vec, r_vec);
    
    py::dict result;
    result["direct"] = res.direct;
    result["indirect"] = res.indirect;
    return result;
}

// 5.c Optimal Policy
py::dict solve_optimal_policy_py(
    // Grid/Param boilerplate
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& c_pol, const std::vector<double>& m_pol, const std::vector<double>& a_pol,
    const std::vector<double>& value, const std::vector<double>& adjust_flag,
    const std::vector<double>& distribution,
    const std::vector<double>& E_Vm, const std::vector<double>& E_V,
    // Optimization Params
    double lambda_y,
    const std::vector<double>& dr_star_path
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    auto income = reconstruct_income(z_grid, Pi_flat);
    auto params = reconstruct_params(beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, m_min);
    auto policy = reconstruct_policy(grid, c_pol, m_pol, a_pol, value, adjust_flag);

    Monad::SsjSolver3D ssj_solver(grid, params, income, policy, distribution, E_Vm, E_V);
    
    int T = dr_star_path.size();
    Monad::OptimalPolicy opt(ssj_solver, T);
    
    Eigen::VectorXd rstar_vec = Eigen::Map<const Eigen::VectorXd>(dr_star_path.data(), T);
    
    auto res = opt.solve_optimal_policy(lambda_y, rstar_vec);
    
    py::dict result;
    result["r_opt"] = res.r_opt;
    result["y_opt"] = res.y_opt;
    result["pi_opt"] = res.pi_opt;
    result["loss"] = res.loss;
    return result;
}

// 5.d MPC Distribution
py::dict compute_mpc_distribution_py(
    // Grid/Param boilerplate
    int Nm, double m_min, double m_max, double m_curv,
    int Na, double a_min, double a_max, double a_curv,
    const std::vector<double>& z_grid, const std::vector<double>& Pi_flat,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    const std::vector<double>& c_pol, const std::vector<double>& m_pol, const std::vector<double>& a_pol,
    const std::vector<double>& value, const std::vector<double>& adjust_flag,
    const std::vector<double>& distribution
) {
    auto grid = reconstruct_grid(Nm, m_min, m_max, m_curv, Na, a_min, a_max, a_curv, z_grid.size());
    // Params needed for MicroAnalyzer? Only policy and grid mostly.
    Monad::TwoAssetParam params; // Dummy
    auto policy = reconstruct_policy(grid, c_pol, m_pol, a_pol, value, adjust_flag);

    Monad::MicroAnalyzer analyzer(grid, policy, params);
    auto stats = analyzer.compute_stats(distribution);
    
    py::dict result;
    result["avg_mpc"] = stats.avg_mpc;
    result["weighted_mpc"] = stats.weighted_mpc;
    result["mpc_by_z"] = stats.mpc_by_z;
    return result;
}

// ============================================================================
// Module Spec
// ============================================================================
PYBIND11_MODULE(monad_core, m) {
    m.doc() = "Monad Engine Core - Extended API v2.1";

    // 1. Household
    m.def("solve_hank_steady_state", &solve_hank_steady_state_py, 
          py::arg("Nm")=50, py::arg("m_min")=-2.0, py::arg("m_max")=50.0, py::arg("m_curv")=3.0,
          py::arg("Na")=40, py::arg("a_min")=0.0, py::arg("a_max")=100.0, py::arg("a_curv")=2.0,
          py::arg("z_grid"), py::arg("Pi_flat"),
          py::arg("beta")=0.97, py::arg("r_m")=0.01, py::arg("r_a")=0.05, py::arg("chi")=20.0, py::arg("sigma")=2.0,
          py::arg("tax_lambda")=0.9, py::arg("tax_tau")=0.15, py::arg("tax_transfer")=0.05,
          py::arg("max_bellman_iter")=1000, py::arg("bellman_tol")=1e-6,
          py::arg("max_dist_iter")=2000, py::arg("dist_tol")=1e-8);
          

          
    m.def("solve_one_asset_steady_state", &solve_one_asset_steady_state_py,
          py::arg("Nm")=50, py::arg("m_min")=-2.0, py::arg("m_max")=50.0, py::arg("m_curv")=3.0,
          py::arg("z_grid"), py::arg("Pi_flat"),
          py::arg("beta")=0.97, py::arg("r_m")=0.02, py::arg("sigma")=2.0,
          py::arg("tax_lambda")=0.9, py::arg("tax_tau")=0.15, py::arg("tax_transfer")=0.05,
          py::arg("max_bellman_iter")=1000, py::arg("bellman_tol")=1e-6,
          py::arg("max_dist_iter")=2000, py::arg("dist_tol")=1e-8);
          
    m.def("probe_policy", &probe_policy_py, "Interp policy at state");
    m.def("compute_mpc_distribution", &compute_mpc_distribution_py, "Compute Agg MPC");
    
    // 2. SSJ
    m.def("compute_jacobians", &compute_jacobians_py);
    m.def("get_transition_matrix", &get_transition_matrix_py, "Export sparse transition matrix");

    // 3. GE & Experiments
    m.def("solve_ge_zlb", &solve_ge_zlb_py, "Solve GE with ZLB");
    m.def("solve_fiscal_shock", &solve_fiscal_shock_py, "Solve Fiscal G/T Shock");
    m.def("decompose_multiplier", &decompose_multiplier_py, "Decompose dY into Direct/Indirect");
    m.def("solve_optimal_policy", &solve_optimal_policy_py, "Solve LQR Optimal Policy");
    
    // 4. Analysis
    m.def("analyze_inequality", &analyze_inequality_py, "Decompose responses by group");

    // ============================================================================
    // 5. Generic Static Solver (Layer 2)
    // ============================================================================
    m.def("solve_static_model", [](py::function py_resid, Eigen::VectorXd guess, 
                                   int max_iter=100, double tol=1e-6, double damping=1.0) {
        
        // Wrap Python function -> C++ std::function
        // Note: This calls back into Python, so GIL is involved.
        // Performance overhead is acceptable for "Evaluation", as detailed in design.
        Monad::ResidualFunc F = [py_resid](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            // Acquire GIL? pybind11 handles this usually if thread states are correct.
            // But usually safer to be careful. py::function call holds GIL.
            py::object result = py_resid(x);
             // Ensure it's converted to Eigen
            return result.cast<Eigen::VectorXd>();
        };
        
        Monad::GenericNewtonSolver::Options opts;
        opts.max_iter = max_iter;
        opts.tol = tol;
        opts.damping_factor = damping;
        opts.verbose = true;
        
        return Monad::GenericNewtonSolver::solve(F, guess, opts);
    }, "Solve static model via C++ Newton, calling back to Python for Residuals");
}

