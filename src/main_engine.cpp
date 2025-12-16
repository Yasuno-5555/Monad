#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <cmath>

// Core Components
#include "Params.hpp"
#include "io/json_loader.hpp"
#include "UnifiedGrid.hpp"
#include "AnalyticalSolver.hpp"
#include "ssj/jacobian_builder.hpp"
#include "ssj/aggregator.hpp"
#include "ssj/ssj_solver.hpp"

// Simple CSV Writer Helpers
void write_csv_ss(const std::string& filename, const UnifiedGrid& grid, 
                  const std::vector<double>& c, const std::vector<double>& a_pol, 
                  const Eigen::VectorXd& D, int nz) {
    std::ofstream f(filename);
    f << "asset,z_idx,consumption,next_asset,distribution\n";
    int na = grid.size;
    
    // c, a_pol, D are flattened (nz * na)
    for(int j=0; j<nz; ++j) {
        for(int i=0; i<na; ++i) {
            int idx = j * na + i;
            if (idx < c.size()) {
                f << grid.nodes[i] << "," << j << "," << c[idx] << "," << a_pol[idx] << "," << D[idx] << "\n";
            }
        }
    }
    f.close();
    std::cout << "[IO] Wrote " << filename << std::endl;
}

void write_csv_trans(const std::string& filename, 
                     const Monad::SsjSolver::TransitionResult& res, int T) {
    std::ofstream f(filename);
    // Export full macro variables
    // dY = Output Gap, dC = Consumption, dN = Labor, dw = Real Wage
    // dpi = Inflation, di = Nominal Rate, dreal_r = Real Rate
    // dB = Debt, dT = Tax Revenue
    f << "period,dY,dC,dN,dw,dpi,di,dreal_r,dB,dT\n";
    for(int t=0; t<T; ++t) {
        f << t << "," 
          << res.dY(t) << "," << res.dC(t) << "," << res.dN(t) << ","
          << res.dw(t) << "," << res.dpi(t) << "," << res.di(t) << ","
          << res.dreal_r(t) << "," << res.dB(t) << "," << res.dT(t) << "\n";
    }
    f.close();
    std::cout << "[IO] Wrote " << filename << std::endl;
}

// Inequality Path Analysis
// Computes consumption/asset distribution metrics along the transition path
void write_inequality_path(const std::string& filename,
                           const UnifiedGrid& grid,
                           const Eigen::VectorXd& D_ss,
                           const std::vector<double>& c_ss,
                           const std::vector<double>& a_pol_ss,
                           const Eigen::VectorXd& dr_path,
                           const Monad::JacobianBuilder::PolicyPartials& partials,
                           const IncomeProcess& income,
                           double r_ss, double w_ss) {
    std::ofstream f(filename);
    f << "period,C_top10,C_bottom50,C_total,A_gini,wealth_top10_share\n";
    
    int na = grid.size;
    int nz = income.n_z;
    int size = na * nz;
    int T = dr_path.size();
    
    // Precompute steady-state asset values for sorting
    std::vector<std::pair<double, int>> asset_idx(size);
    for(int j=0; j<nz; ++j) {
        for(int i=0; i<na; ++i) {
            int idx = j*na + i;
            asset_idx[idx] = {grid.nodes[i], idx};
        }
    }
    std::sort(asset_idx.begin(), asset_idx.end());
    
    // Find cutoff indices for bottom 50% and top 10%
    double cum_mass = 0.0;
    int bottom50_cutoff = 0;
    int top10_cutoff = size - 1;
    
    for(int k=0; k<size; ++k) {
        cum_mass += D_ss[asset_idx[k].second];
        if(cum_mass < 0.50) bottom50_cutoff = k;
        if(cum_mass < 0.90) top10_cutoff = k;
    }
    
    // Compute metrics at each time step
    for(int t=0; t<T; ++t) {
        double dr_t = dr_path[t];
        
        // Linear approximation: dc_t = dc/dr * dr_t
        // (Ignoring dw effects for simplicity - they're second order)
        double C_bottom50 = 0.0;
        double C_top10 = 0.0;
        double C_total = 0.0;
        double A_total = 0.0;
        double A_top10 = 0.0;
        
        for(int k=0; k<size; ++k) {
            int idx = asset_idx[k].second;
            double mass = D_ss[idx];
            
            // Consumption at time t (linear approx)
            double c_t = c_ss[idx] + partials.dc_dr[idx] * dr_t;
            double a_t = grid.nodes[idx % na]; // Current assets
            
            C_total += c_t * mass;
            A_total += a_t * mass;
            
            if(k <= bottom50_cutoff) {
                C_bottom50 += c_t * mass;
            }
            if(k > top10_cutoff) {
                C_top10 += c_t * mass;
                A_top10 += a_t * mass;
            }
        }
        
        // Gini coefficient (simplified: using asset distribution)
        // Gini = 1 - 2 * integral of Lorenz curve
        double gini = 0.0;
        double lorenz_area = 0.0;
        double cum_pop = 0.0;
        double cum_wealth = 0.0;
        
        for(int k=0; k<size; ++k) {
            int idx = asset_idx[k].second;
            double mass = D_ss[idx];
            double a = grid.nodes[idx % na];
            
            double prev_cum_pop = cum_pop;
            double prev_cum_wealth = cum_wealth;
            
            cum_pop += mass;
            cum_wealth += a * mass / A_total;
            
            // Trapezoidal area under Lorenz curve
            lorenz_area += 0.5 * (prev_cum_wealth + cum_wealth) * mass;
        }
        gini = 1.0 - 2.0 * lorenz_area;
        
        double top10_share = A_top10 / A_total;
        
        f << t << "," << C_top10 << "," << C_bottom50 << "," << C_total 
          << "," << gini << "," << top10_share << "\n";
    }
    
    f.close();
    std::cout << "[IO] Wrote " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // 1. Setup & Load Config
        std::string config_path = (argc > 1) ? argv[1] : "model_ir.json";
        std::cout << "=== Monad Engine v1.1 ===" << std::endl;
        std::cout << "Config: " << config_path << std::endl;

        if (!std::filesystem::exists(config_path)) {
            std::cerr << "Error: Config file not found: " << config_path << std::endl;
            return 1;
        }

        UnifiedGrid grid;
        MonadParams params;
        
        // Load Grid & Params
        JsonLoader::load_model(config_path, grid, params);

        // 2. Solve Steady State
        std::cout << "\n--- Step 1: Solving Steady State ---" << std::endl;
        
        double r_guess = params.get("r_guess", 0.02);
        
        // Run Analytical Solver
        AnalyticalSolver::solve_steady_state(grid, r_guess, params);
        
        double r_ss = r_guess;
        double beta = params.get_required("beta");
        double sigma = params.get("sigma", 2.0);
        double alpha = params.get("alpha", 0.33); 
        double A = params.get("A", 1.0);
        
        double K_dem = std::pow(r_ss / (alpha * A), 1.0 / (alpha - 1.0));
        double w_ss = (1.0 - alpha) * A * std::pow(K_dem, alpha);
        
        std::cout << "  -> Equilibrium r = " << r_ss << " (" << r_ss * 100.0 << "%)" << std::endl;

        // Retrieve Full Steady State Policy & Distribution
        // NOTE: Implemented helper in AnalyticalSolver to retrieve these after convergence
        std::vector<double> c_ss, mu_ss, a_pol_ss, D_ss_vec;
        AnalyticalSolver::get_steady_state_policy(grid, r_ss, params, c_ss, mu_ss, a_pol_ss, D_ss_vec);

        // Convert D_ss_vec to Eigen for compatibility (though we really use the vec for CSV now)
        // Since D_ss is used in CSV writer which I updated to take Eigen::VectorXd, I'll map it.
        // Wait, write_csv_ss takes Eigen::VectorXd& D.
        Eigen::VectorXd D_ss = Eigen::Map<Eigen::VectorXd>(D_ss_vec.data(), D_ss_vec.size());

        // Output Steady State
        write_csv_ss("steady_state.csv", grid, c_ss, a_pol_ss, D_ss, params.income.n_z);

        // 3. Prepare for SSJ (Partials)
        // Enable SSJ for 2D
        std::cout << "\n--- Step 2: Building 2D Transition Matrix (SSJ) ---" << std::endl;
        
        // Build 2D Transition Matrix
        auto Lambda_ss = Monad::JacobianBuilder::build_transition_matrix_2d(a_pol_ss, grid, params.income);
        
        // Check stationarity: Lambda^T * D_ss should be D_ss
        Eigen::VectorXd D_check = Lambda_ss.transpose() * D_ss;
        double err = (D_check - D_ss).norm();
        std::cout << "[SSJ] Stationarity Check Error: " << err << std::endl;

        std::cout << "\n--- Step 3: Computing 2D Partials (Jacobian) ---" << std::endl;
        auto partials = Monad::JacobianBuilder::compute_partials_2d(
            grid, params, mu_ss, r_ss, w_ss
        );
        
        std::cout << "[SSJ] Partial da/dr norm: " << partials.da_dr.norm() << std::endl;
        std::cout << "[SSJ] Partial da/dw norm: " << partials.da_dw.norm() << std::endl;
        
        std::cout << "[INFO] SSJ Framework (Lambda and Partials) built successfully." << std::endl;
 
        // 4. Check for Shocks & Solve Transition
        bool run_shock = true; 
        
        if (run_shock) {
            std::cout << "\n--- Step 3: Solving NK Transition Path (SSJ) ---" << std::endl;
            int T = 200;
            
            // NK Solution: Output Gap Response to Monetary Policy Shock
            // Assumption: c_ss captured from get_steady_state_policy as c_ss
            
            // v1.7: Check for wage stickiness and Taylor Rule tuning
            double theta_w = params.get("theta_w", 0.75); // Default to Sticky (v1.7)
            double phi_pi = params.get("phi_pi", 1.5);    // Default Taylor Rule
            
            // Solve Linear System (using NK+Wage solver)
            auto res = Monad::SsjSolver::solve_nk_wage_transition(
                T, grid, D_ss, Lambda_ss, a_pol_ss, c_ss, partials, params.income, r_ss, theta_w, phi_pi
            );
            
            // Output Results (Full Macro Paths)
            write_csv_trans("transition_nk.csv", res, T);
            
            // Phase 3: Inequality Analysis
            std::cout << "\n--- Step 4: Computing Inequality Path ---" << std::endl;
            write_inequality_path("inequality_path.csv", grid, D_ss, c_ss, a_pol_ss,
                                  res.dY, partials, params.income, r_ss, w_ss);
        }

        std::cout << "\n=== Finished Successfully ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
