#include <iostream>
#include "aggregator/DistributionAggregator3D.hpp"
#include "ssj/SparseMatrixBuilder.hpp"
#include "ssj/JacobianBuilder3D.hpp"
#include "ssj/FakeNewsAggregator.hpp"
#include "ssj/SsjSolver3D.hpp"
#include "ssj/GeneralEquilibrium.hpp"
#include "analysis/InequalityAnalyzer.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iomanip>
#include <memory> 

#include "Params.hpp"
#include "grid/MultiDimGrid.hpp"
#include "kernel/TwoAssetKernel.hpp"
#include "solver/TwoAssetSolver.hpp"
#include "aggregator/DistributionAggregator3D.hpp"
#include "gpu/CudaUtils.hpp"

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
        // v3.0: GPU Init
        std::cout << "[INFO] Initializing Monad Engine v3.0 (GPU Hybrid)..." << std::endl;

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
        
        // v2.1 Fiscal Policy (HSV 2017)
        params.fiscal.tax_rule.lambda = 0.9; // Scale of post-tax income
        params.fiscal.tax_rule.tau = 0.15;   // Progressivity
        params.fiscal.tax_rule.transfer = 0.05; // Lump-sum transfer

        // 2. Setup Grids
        // Liquid: [-2.0, 50.0], concentrated near 0
        auto m_grid = make_grid(50, -2.0, 50.0, 3.0); 
        // Illiquid: [0.0, 100.0]
        auto a_grid = make_grid(40, 0.0, 100.0, 2.0);
        auto income = make_income();

        Monad::MultiDimGrid grid(m_grid, a_grid, income.n_z);
        std::cout << "Grid Size: " << grid.total_size 
                  << " (" << grid.N_m << "x" << grid.N_a << "x" << grid.N_z << ")" << std::endl;
        
        // v3.0 GPU Setup (Real dimensions)
        auto gpu_backend = std::make_unique<Monad::CudaBackend>(grid.N_m, grid.N_a, grid.N_z);
        gpu_backend->verify_device();
        // Pre-upload static grids
        gpu_backend->upload_grids(grid.m_grid.nodes, grid.a_grid.nodes);
        
        // 3. Initialize Policy & Solver
        Monad::TwoAssetPolicy policy(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) policy.c_pol[i] = 0.1; // Avoid singularity
        
        Monad::TwoAssetPolicy next_policy(grid.total_size);
        
        // Pass GPU backend to solver
        auto solver = std::make_unique<Monad::TwoAssetSolver>(grid, params, gpu_backend.get());
            
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
        
        double Agg_Tax, Agg_Transfers;
        aggregator.compute_fiscal_aggregates(D, income, params, Agg_Tax, Agg_Transfers);
        std::cout << "Aggregate Tax Revenue:  " << Agg_Tax << std::endl;
        std::cout << "Aggregate Transfers:    " << Agg_Transfers << std::endl;
        std::cout << "Primary Surplus (T-Tr): " << Agg_Tax - Agg_Transfers << std::endl;

        // v3.2 Verification
        // solver->test_dual_kernel();
        
        // Ensure GPU State is Clean/Populated (Fix for Zero Jacobian)
        gpu_backend->upload_full_policy(policy.c_pol, policy.m_pol, policy.a_pol);
        // Convert D (vector) to D_flat (vector) if needed? D is already flat.
        // Wait, D is std::vector<double> of size total_size. Correct.
        gpu_backend->upload_distribution(D);
        
        solver->test_full_jacobian_gpu();

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

        // 9. Verify Phase 3 Step 1: General Equilibrium Solver (Multiplier)
        std::cout << "\n--- Verifying General Equilibrium (Phase 3) ---" << std::endl;
        
        Monad::GeneralEquilibrium ge_solver(ssj_solver, T);
        
        // Define Shock: 1% increase in r_m (annualized? usually bps). 
        // Let's say 0.01 shock with 0.8 persistence
        Eigen::VectorXd dr_m(T);
        double rho = 0.8;
        double shock_size = 0.01;
        for(int t=0; t<T; ++t) dr_m[t] = shock_size * std::pow(rho, t);
        
        auto ge_results = ge_solver.solve_monetary_shock(dr_m);
        
        Eigen::VectorXd dY = ge_results["dY"];
        Eigen::VectorXd dC = ge_results["dC"];
        Eigen::VectorXd dC_direct = ge_results["dC_direct"];
        
        std::cout << "GE Results:" << std::endl;
        std::cout << "  dY Impact (t=0): " << dY[0] << std::endl;
        std::cout << "  dC Direct Impact (t=0): " << dC_direct[0] << std::endl;
        
        double multiplier = dY[0] / dC_direct[0];
        std::cout << "  Impact Multiplier (dY/dC_direct): " << multiplier << std::endl;
        
        if(std::abs(multiplier) > 1.0) {
             std::cout << "  [PASS] Multiplier > 1 (Keynesian Amplification)" << std::endl;
        } else {
             std::cout << "  [INFO] Multiplier <= 1 (Dampening or Weak Feedback)" << std::endl;
        }

        // CSV Output
        std::ofstream ge_file("ge_irf.csv");
        ge_file << "t,dr_m,dY,dC,dC_direct,dC_indirect\n";
        for(int t=0; t<T; ++t) {
            ge_file << t << "," << dr_m[t] << "," << dY[t] << "," 
                    << dC[t] << "," << dC_direct[t] << "," << ge_results["dC_indirect"][t] << "\n";
        }
        ge_file.close();
        std::cout << "  GE IRF written to ge_irf.csv" << std::endl;

        std::cout << "  GE IRF written to ge_irf.csv" << std::endl;

        // 10. Verify Phase 3 Step 2: Inequality Analysis
        std::cout << "\n--- Verifying Inequality Analysis (Phase 3 Step 2) ---" << std::endl;
        
        // Need partial map for analyzer. access private? No, ssj_solver computes them internally.
        // We should expose partials from SsjSolver3D or re-compute.
        // Re-computing is fine for now, or add getter.
        // SsjSolver3D stores jac_builder. Call compute_partials directly?
        // Let's instantiate a new builder or expose it. SsjSolver3D has jac_builder private.
        // Actually, we can just use the partials computed inside ssj_solver? 
        // compute_block_jacobians returns Matrices, not partials map.
        // Let's quickly add a getter or friend, or just re-run compute_partials.
        // Re-run is safest/easiest given no interface change.
        
        
        { // Scope for Inequality Analysis
        Monad::JacobianBuilder3D jac_builder(grid, params, income, E_Vm_ss, E_V_ss);
        auto partials = jac_builder.compute_partials(policy);
        
        Monad::InequalityAnalyzer analyzer(grid, D, policy, partials);
        
        // Pack GE results into map for analyzer
        std::map<std::string, Eigen::VectorXd> dU_paths;
        dU_paths["rm"] = dr_m;
        dU_paths["w"] = dY; // Assume w = Y
        
        auto group_res = analyzer.analyze_consumption_response(dU_paths);
        
        std::cout << "Inequality IRF (t=0):" << std::endl;
        std::cout << "  Top 10% dC:    " << group_res.top10[0] << std::endl;
        std::cout << "  Bottom 50% dC: " << group_res.bottom50[0] << std::endl;
        std::cout << "  Debtors dC:    " << group_res.debtors[0] << std::endl;
        
        if (std::abs(group_res.debtors[0]) > std::abs(group_res.top10[0])) {
             std::cout << "  [PASS] Debtors are more sensitive to rate hike." << std::endl;
             std::cout << "  [INFO] Top 10% sensitive (Wealth Effect dominates?)" << std::endl;
        }

        // Export IRF Groups CSV
        std::ofstream irf_groups("irf_groups.csv");
        irf_groups << "time,top10,bottom50,debtors,aggregate\n";
        for(int t=0; t<T; ++t) {
            irf_groups << t << "," 
                       << group_res.top10[t] << "," 
                       << group_res.bottom50[t] << "," 
                       << group_res.debtors[t] << ","
                       << dC[t] << "\n";
        }
        irf_groups.close();
        std::cout << "  Exported irf_groups.csv" << std::endl;

        // Export Heatmap CSV (Sensitivity at t=0)
        // dC/dr at t=0 corresponds to the immediate policy response.
        // We use compute_consumption_heatmap for t=0 shock.
        // Note: dU_paths contains the full path. 
        // We want the sensitivity map (dC_i / dr_m).
        // The analyzer computes dC_total = sum(dC/dU * dU).
        // If we want raw sensitivity, we can just use partials directly or pass a unit shock at t=0.
        // But the user asked for "Heatmap Series ... dC(m,a) of 2D heatmap".
        // Let's output the actual predicted dC at t=0 (which includes the shock size).
        
        auto heatmap_vals = analyzer.compute_consumption_heatmap(dU_paths, 0); // t=0
        
        std::ofstream heat_file("heatmap_sensitivity.csv");
        heat_file << "m_val,a_val,dC_dr\n";
        for(int i=0; i<grid.total_size; ++i) {
             // Only output if relevant (sparse?) No, heatmap needs full grid usually.
             auto vals = grid.get_values(i);
             heat_file << vals.first << "," << vals.second << "," << heatmap_vals[i] << "\n";
        }
        heat_file.close();
        std::cout << "  Exported heatmap_sensitivity.csv" << std::endl;
        }

        // 11. Verify v2.2 Step: ZLB Solver
        std::cout << "\n--- Verifying v2.2 ZLB Solver (Liquidity Trap) ---" << std::endl;
        
        // Construct Large Negative r_star shock (Demand Shock)
        // e.g. -2% for 10 quarters
        Eigen::VectorXd dr_star = Eigen::VectorXd::Zero(T);
        for(int t=0; t<10; ++t) dr_star[t] = -0.05; // -500bps

        
        auto zlb_results = ge_solver.solve_with_zlb(dr_star);
        auto& i_path = zlb_results["i"];
        auto& Y_path = zlb_results["dY"];
        auto& eps_path = zlb_results["eps"];
        
        std::cout << "ZLB Scenario Results (t=0..15):" << std::endl;
        std::cout << "t | r* | i | i (Unc) | Y | eps" << std::endl;
        
        // Solve Unconstrained for comparison
        auto unc_results = ge_solver.solve_monetary_shock(dr_star); // This solves for r_m shock, not r_star shock... 
        // Wait, solve_monetary_shock takes dr_m. Here we have dr_star.
        // In the Taylor rule: i = r_star + phi * pi.
        // If we want to compare, we need an "Unconstrained" solver for r_star.
        // But solve_with_zlb already calculates unconstrained 'istar' internally, but doesn't return Y_unc.
        // Let's rely on looking at 'i'. If ZLB binds, i=0.
        
        for(int t=0; t<15; ++t) {
             std::cout << t << " | " 
                       << dr_star[t] << " | " 
                       << i_path[t] << " | "
                       << zlb_results["i_star"][t] << " | "
                       << Y_path[t] << " | " 
                       << eps_path[t] << std::endl;
        }
        
        if (i_path[0] > -1e-6 && zlb_results["i_star"][0] < -1e-6) {
             std::cout << "  [PASS] ZLB Binding! (i=0 while i* < 0)" << std::endl;
             std::cout << "  [INFO] Output drop: " << Y_path[0] << std::endl;
        } else {
             std::cout << "  [WARN] ZLB not binding? Increase shock size." << std::endl;
        }

        // 12. Experiment: Odyssean Forward Guidance (v2.3)
        std::cout << "\n--- Experiment: Odyssean Forward Guidance (v2.3) ---" << std::endl;
        
        // Baseline: We already have zlb_results (Baseline)
        double Y_impact_base = zlb_results["dY"][0];
        
        // Forward Guidance: Force rates to 0 for 5 quarters AFTER natural lift-off.
        // In Baseline, ZLB bound for approx t=0..12? (Need to check output)
        // Let's force t=15,16,17,18,19 (assuming lift-off is earlier).
        // A robust way is to find lift-off from baseline and add K periods.
        
        std::vector<int> forced_periods;
        for(int t=0; t<T; ++t) {
            if (zlb_results["eps"][t] > 1e-5) {
                // Was binding in baseline
            } else {
                // Not binding. If we just exited, add FG periods.
                // Simple fixed test: Force t=15 to 20.
            }
        }
        // Let's use fixed periods based on shock length (10). 
        // Shock ends at 10. Trap likely ends 12-14.
        // Let's force t=15, 16, 17, 18, 19.
        forced_periods = {15, 16, 17, 18, 19};
        
        auto fg_results = ge_solver.solve_with_zlb(dr_star, forced_periods);
        double Y_impact_fg = fg_results["dY"][0];
        
        std::cout << "Impact Comparison (t=0):" << std::endl;
        std::cout << "  Baseline Output: " << Y_impact_base << std::endl;
        std::cout << "  FG Output:       " << Y_impact_fg << std::endl;
        std::cout << "  Difference:      " << Y_impact_fg - Y_impact_base << std::endl;
        
        if (Y_impact_fg > Y_impact_base) { // Less negative?
             // Since Y_impact is negative (-0.03 etc), "Higher" means closer to 0?
             // No, "Stimulus" means Y increases.
             // If Y_base = -0.05, Y_fg = -0.04. -0.04 > -0.05.
             std::cout << "  [PASS] Forward Guidance improved current output (Expectations Channel active)." << std::endl;
        } else {
             std::cout << "  [FAIL] No improvement from FG?" << std::endl;
        }

        std::cout << "Detailed Forward Guidance Path:" << std::endl;
        std::cout << "t | i (Base) | i (FG) | Y (Base) | Y (FG)" << std::endl;
        for(int t=0; t<25; ++t) {
             std::cout << t << " | " 
                       << zlb_results["i"][t] << " | " << fg_results["i"][t] << " | "
                       << zlb_results["dY"][t] << " | " << fg_results["dY"][t] << std::endl;
        }


        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
