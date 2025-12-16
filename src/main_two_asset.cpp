#include <iostream>
#include "aggregator/DistributionAggregator3D.hpp"
#include "ssj/SparseMatrixBuilder.hpp"
#include "ssj/JacobianBuilder3D.hpp"
#include "ssj/FakeNewsAggregator.hpp"
#include "ssj/SsjSolver3D.hpp"
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <memory> 

#include "Params.hpp"
#include "grid/MultiDimGrid.hpp"
#include "kernel/TwoAssetKernel.hpp"
#include "solver/TwoAssetSolver.hpp"
#include "aggregator/DistributionAggregator3D.hpp"

// Helper: Power Grid Generator (concentration near min)
UnifiedGrid make_grid(int size, double min, double max, double curv) {
    UnifiedGrid g;
    g.resize(size);
    for(int i=0; i<size; ++i) {
        double t = (double)i / (size - 1);
        g.nodes[i] = min + (max - min) * std::pow(t, curv);
    }
    return g;
}

// Helper: Simple Income Process (Example)
IncomeProcess make_income() {
    // 2-state simple process for testing
    // z = [0.8, 1.2], Pi = [[0.9, 0.1], [0.1, 0.9]]
    IncomeProcess p;
    p.n_z = 2;
    p.z_grid = {0.8, 1.2};
    p.Pi_flat = {0.9, 0.1, 
                 0.1, 0.9};
    // Stationary dist: [0.5, 0.5]
    return p;
}

int main() {
    try {
        std::cout << "=== Monad Engine v2.0: Two-Asset Stationary Solver ===" << std::endl;

        // 1. Setup Parameters
        Monad::TwoAssetParam params;
        params.beta = 0.97;
        params.r_m = 0.01;  // Liquid Rate (Low)
        params.r_a = 0.05;  // Illiquid Rate (High)
        params.chi = 20.0;  // Adjustment Cost Scale (Moderate)
        params.sigma = 2.0; // CRRA
        params.m_min = -2.0; // Borrowing limit matching grid min
        // params.chi = 1e8; // Uncomment to test Liquid-Only Limit

        // 2. Setup Grids
        // Liquid: [-2.0, 50.0], concentrated near 0
        auto m_grid = make_grid(50, -2.0, 50.0, 3.0); 
        // Illiquid: [0.0, 100.0]
        auto a_grid = make_grid(40, 0.0, 100.0, 2.0);
        auto income = make_income();

        Monad::MultiDimGrid grid(m_grid, a_grid, income.n_z);
        std::cout << "Grid Size: " << grid.total_size 
                  << " (" << grid.N_m << "x" << grid.N_a << "x" << grid.N_z << ")" << std::endl;
        
        // 3. Initialize Policy & Solver
        Monad::TwoAssetPolicy policy(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) policy.c_pol[i] = 0.1; // Avoid singularity
        
        Monad::TwoAssetPolicy next_policy(grid.total_size);
        
        auto solver = std::make_unique<Monad::TwoAssetSolver>(grid, params);
            
            // 4. Policy Iteration (VFI / EGM)
            std::cout << "\n--- Solving Policy ---" << std::endl;
        for(int iter=0; iter<1000; ++iter) {
            double diff = solver->solve_bellman(policy, next_policy, income);
            // Update
            policy = next_policy; // Copy

            if(iter % 10 == 0) {
                std::cout << "Iter " << iter << " Diff: " << diff << std::endl;
            }
            if(diff < 1e-6) {
                std::cout << "Converged at iter " << iter << std::endl;
                break;
            }
        }

        // 5. Distribution Iteration
        std::cout << "\n--- Solving Distribution ---" << std::endl;
        Monad::DistributionAggregator3D aggregator(grid);
        std::vector<double> D = aggregator.init_uniform();
        std::vector<double> D_next(grid.total_size);

        for(int iter=0; iter<2000; ++iter) {
            double diff = aggregator.forward_iterate(D, D_next, policy, income);
            D = D_next; // Copy

            if(iter % 50 == 0) std::cout << "Dist Iter " << iter << " Diff: " << diff << std::endl;
            if(diff < 1e-8) {
                std::cout << "Distribution Converged at iter " << iter << std::endl;
                break;
            }
        }

        // 6. Compute Aggregates
        double Agg_Liquid, Agg_Illiquid;
        aggregator.compute_aggregates(D, Agg_Liquid, Agg_Illiquid);
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Aggregate Liquid (B):   " << Agg_Liquid << std::endl;
        std::cout << "Aggregate Illiquid (K): " << Agg_Illiquid << std::endl;
        std::cout << "Total Wealth:           " << Agg_Liquid + Agg_Illiquid << std::endl;

        // 7. Export Data for Visualization
        std::cout << "\nWriting output files..." << std::endl;
        
        // Policy Output
        std::ofstream fp("policy_2asset.csv");
        fp << "m_idx,a_idx,z_idx,m_val,a_val,z_val,c,m_prime,a_prime,adjust_flag\n";
        for(int i=0; i<grid.total_size; ++i) {
            int im, ia, iz;
            grid.get_coords(i, im, ia, iz);
            auto vals = grid.get_values(i);
            
            fp << im << "," << ia << "," << iz << ","
               << vals.first << "," << vals.second << "," << income.z_grid[iz] << ","
               << policy.c_pol[i] << "," << policy.m_pol[i] << "," 
               << policy.a_pol[i] << "," << policy.adjust_flag[i] << "\n";
        }
        fp.close();

        // Distribution Output
        std::ofstream fd("dist_2asset.csv");
        fd << "m_val,a_val,z_val,mass\n";
        for(int i=0; i<grid.total_size; ++i) {
            if(D[i] > 1e-9) { // Sparse output
                auto vals = grid.get_values(i);
                int im, ia, iz;
                grid.get_coords(i, im, ia, iz);
                fd << vals.first << "," << vals.second << "," << income.z_grid[iz] << "," << D[i] << "\n";
            }
        }
        fd.close();

        // 6. Verify Sparse Matrix Consistency (Phase 2 Step 1)
        std::cout << "\n--- Verifying Sparse Transition Matrix (Lambda) ---" << std::endl;
        Monad::SparseMatrixBuilder sparse_builder(grid, income);
        auto Lambda = sparse_builder.build_transition_matrix(policy);
        
        std::cout << "Lambda built. Non-zeros: " << Lambda.nonZeros() << std::endl;
        
        // Convert std::vector dist to Eigen::VectorXd
        Eigen::VectorXd D_ss(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) D_ss[i] = D[i];

        // Check Stationarity: D = Lambda * D
        Eigen::VectorXd D_next_sparse = Lambda * D_ss;
        double error_norm = (D_next_sparse - D_ss).norm();
        double error_max = (D_next_sparse - D_ss).lpNorm<Eigen::Infinity>();
        
        std::cout << "Stationarity Check:" << std::endl;
        std::cout << "  L2 Norm Error: " << error_norm << std::endl;
        std::cout << "  Max Error:     " << error_max << std::endl;
        
        if(error_max < 1e-7) {
             std::cout << "  [PASS] Sparse Matrix matches Forward Iteration (Tol 1e-7)." << std::endl;
        } else {
             std::cout << "  [FAIL] Discrepancy detected!" << std::endl;
        }

        // 7. Verify Phase 2 Step 2: Dual EGM & Fake News
        std::cout << "\n--- Verifying Dual EGM & Fake News ---" << std::endl;
        
        // Need SS Expectations E_Vm and E_V
        // In TwoAssetSolver, these are stored in `solver->E_Vm_next`.
        // But `solver` processes internally and we don't expose them.
        // Quick Fix: We need to re-extract them or assume solver state is valid.
        // Wait, solver->solve_bellman updates `solver->E_Vm_next` at the end of iteration?
        // No, `compute_expectations` is called, updating E_V, E_Vm.
        // But solver is unique_ptr. Is E_Vm_next public? No, usually private.
        // Assuming we need to add a getter to TwoAssetSolver or make friends.
        
        // Let's modify TwoAssetSolver.hpp to make E_Vm_next public or add getter.
        // For now, I will use a placeholder or hack.
        // Better: Add getter to TwoAssetSolver.hpp now.
        
        // Assume getter exists: solver->E_Vm_next, solver->E_V_next
        
        const auto& E_Vm_ss = solver->E_Vm_next; 
        const auto& E_V_ss  = solver->E_V_next;

        Monad::JacobianBuilder3D jac_builder(grid, params, income, E_Vm_ss, E_V_ss);
        auto partials = jac_builder.compute_partials(policy);
        
        std::cout << "Computed Partials for 'rm':" << std::endl;
        // Check non-zero
        double sum_dc = 0.0;
        double sum_dm = 0.0;
        for(double v : partials["c"]["rm"]) sum_dc += std::abs(v);
        for(double v : partials["m"]["rm"]) sum_dm += std::abs(v);
        
        std::cout << "  Sum |dc/drm|: " << sum_dc << std::endl;
        std::cout << "  Sum |dm'/drm|: " << sum_dm << std::endl;
        
        if(sum_dc > 1e-6) std::cout << "  [PASS] c responds to r_m." << std::endl;
        else              std::cout << "  [FAIL] c unresponsive?" << std::endl;

        // Fake News
        Monad::FakeNewsAggregator fn_agg(grid, income, Lambda);
        auto F_rm = fn_agg.compute_fake_news_vector(partials["m"]["rm"], partials["a"]["rm"], policy, D);
        
        double sum_F = 0.0;
        for(double v : F_rm) sum_F += v;
        std::cout << "  Sum F_rm (Mass Drift): " << sum_F << " (Should be ~0)" << std::endl;
        
        // 8. Verify Phase 2 Step 3: Sequence Space Solver (IRFs)
        std::cout << "\n--- Verifying Sequence Space Solver (IRF) ---" << std::endl;
        
        Monad::SsjSolver3D ssj_solver(grid, params, income, policy, D, E_Vm_ss, E_V_ss);
        int T = 100;
        auto jacobians = ssj_solver.compute_block_jacobians(T);
        
        std::cout << "Computed Jacobians (T=" << T << "):" << std::endl;
        if(jacobians.count("C") && jacobians["C"].count("rm")) {
            Eigen::MatrixXd J_C_rm = jacobians["C"]["rm"];
            
            // Check Impact and Persistence
            std::cout << "  dC/drm Impact (t=0): " << J_C_rm(0, 0) << std::endl;
            std::cout << "  dC/drm (t=0, shock at t=0): " << J_C_rm(0, 0) << " (Direct + Dist)" << std::endl;
            std::cout << "  dC/drm (t=10, shock at t=0): " << J_C_rm(10, 0) << " (Persistence)" << std::endl;
            
            // Verify Toeplitz structure (J(t,s) approx J(t-k, s-k))
            double toeplitz_error = std::abs(J_C_rm(5, 5) - J_C_rm(0, 0));
            std::cout << "  Toeplitz Check (Diag 5 vs 0): " << toeplitz_error << std::endl;
            
            if(std::abs(J_C_rm(0, 0)) > 1e-6) std::cout << "  [PASS] Consumption responds to r_m." << std::endl;
            else std::cout << "  [FAIL] Zero response?" << std::endl;
            
            // Output IRF to file
            std::ofstream irf_file("irf_test.csv");
            irf_file << "t,dC_rm\n";
            for(int t=0; t<T; ++t) {
                irf_file << t << "," << J_C_rm(t, 0) << "\n";
            }
            irf_file.close();
            std::cout << "  IRF written to irf_test.csv" << std::endl;
        } else {
            std::cout << "  [FAIL] Missing Jacobian Block C-rm" << std::endl;
        }

        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
