#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>


#include "Params.hpp"
#include "grid/MultiDimGrid.hpp"
#include "kernel/TwoAssetKernel.hpp"
#include "solver/TwoAssetSolver.hpp"
#include "aggregator/DistributionAggregator3D.hpp"
#include "ssj/SsjSolver3D.hpp"
#include "ssj/GeneralEquilibrium.hpp"
#include "gpu/CudaUtils.hpp"

// Helper: Power Grid Generator (concentration near min)
Monad::UnifiedGrid make_grid(int size, double min, double max, double curv) {
    Monad::UnifiedGrid g;
    g.resize(size);
    for(int i=0; i<size; ++i) {
        double t = (double)i / (size - 1);
        g.nodes[i] = min + (max - min) * std::pow(t, curv);
    }
    return g;
}

// Helper: Simple Income Process
Monad::IncomeProcess make_income() {
    Monad::IncomeProcess p;
    p.n_z = 2;
    p.z_grid = {0.8, 1.2};
    p.Pi_flat = {0.9, 0.1, 0.1, 0.9};
    return p;
}

// Helper to write matrix to CSV
void write_matrix_csv(const Eigen::MatrixXd& mat, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }
    
    // Header (optional? Let's just write raw matrix)
    // No, let's write rows
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j < mat.cols() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "  Exported " << filename << " (" << mat.rows() << "x" << mat.cols() << ")" << std::endl;
}

int run_experiment_tt(const std::string& config_path) {
    try {
        std::cout << "[INFO] Monad Lab: Full TT Jacobian Experiment" << std::endl;
        std::cout << "Loading Config: " << config_path << std::endl;
        // In full refactor, we would load params here. Keeping hardcoded as demo fallback
        // but acknowledging config path.

        // 1. Setup Parameters
        Monad::TwoAssetParam params;
        params.beta = 0.97;
        params.r_m = 0.01;
        params.r_a = 0.05;
        params.chi = 20.0;
        params.sigma = 2.0; 
        params.m_min = -2.0;
        
        // Fiscal
        params.fiscal.tax_rule.tau = 0.15;
        params.fiscal.tax_rule.transfer = 0.05;

        // 2. Setup Grids
        auto m_grid = make_grid(50, -2.0, 50.0, 3.0); 
        auto a_grid = make_grid(40, 0.0, 100.0, 2.0);
        auto income = make_income();

        Monad::MultiDimGrid grid(m_grid, a_grid, income.n_z);
        
        // GPU
        auto gpu_backend = std::make_unique<Monad::CudaBackend>(grid.N_m, grid.N_a, grid.N_z);
        gpu_backend->verify_device();
        gpu_backend->upload_grids(grid.m_grid.nodes, grid.a_grid.nodes);
        
        // 3. Initialize & Solve Steady State
        Monad::TwoAssetPolicy policy(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) policy.c_pol[i] = 0.1; 
        Monad::TwoAssetPolicy next_policy(grid.total_size);
        
        auto solver = std::make_unique<Monad::TwoAssetSolver>(grid, params, gpu_backend.get());
            
        std::cout << "Solving SS Policy..." << std::endl;
        for(int iter=0; iter<2000; ++iter) {
            double diff = solver->solve_bellman(policy, next_policy, income);
            policy = next_policy; 
            if(diff < 1e-7) break;
        }

        std::cout << "Solving SS Distribution..." << std::endl;
        Monad::DistributionAggregator3D aggregator(grid);
        std::vector<double> D = aggregator.init_uniform();
        std::vector<double> D_next(grid.total_size);

        for(int iter=0; iter<3000; ++iter) {
            double diff = aggregator.forward_iterate(D, D_next, policy, income);
            D = D_next;
            if(diff < 1e-9) break;
        }

        // 4. Compute Full Reference Jacobians
        std::cout << "Computing Sequence Space Jacobians (Full TT)..." << std::endl;
        
        // Need Expectations
        const auto& E_Vm_ss = solver->E_Vm_next; 
        const auto& E_V_ss  = solver->E_V_next;

        Monad::SsjSolver3D ssj_solver(grid, params, income, policy, D, E_Vm_ss, E_V_ss);
        
        int T = 80; // Horizon
        auto jacobians = ssj_solver.compute_block_jacobians(T);
        
        // 5. Export TT Jacobians
        // Structure: jacobians[Output][Input]
        // Outputs: C (Consumption), B (Liquid Savings), K (Illiquid Assets)
        // Inputs: rm, ra, w
        
        std::vector<std::string> outputs = {"C", "B", "K"};
        std::vector<std::string> inputs  = {"rm", "ra", "w"};
        
        for(const auto& out : outputs) {
            for(const auto& inp : inputs) {
                if(jacobians.count(out) && jacobians[out].count(inp)) {
                    std::string fname = "tt_jacobian_" + out + "_" + inp + ".csv";
                    write_matrix_csv(jacobians[out][inp], fname);
                } else {
                    std::cout << "  [WARN] Missing block " << out << " - " << inp << std::endl;
                }
            }
        }
        
        std::cout << "Done. All matrices exported." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
