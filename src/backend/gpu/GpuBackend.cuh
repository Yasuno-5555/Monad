#pragma once

#ifdef MONAD_GPU
#include <cuda_runtime.h>
#endif
#include <vector>

namespace Monad {

// Forward declare types needed by GPU backend
// These avoid pulling in Eigen or complex headers
struct GpuGridSizes {
    int N_m;
    int N_a;
    int N_h;
    int N_z;
    int total_size;
};

class GpuBackend {
    GpuGridSizes grid_sizes;
    
    // Device pointers
    double* d_V = nullptr;
    double* d_V_next = nullptr;
    double* d_c = nullptr;
    double* d_c_out = nullptr;  
    double* d_m_pol = nullptr;
    double* d_a_pol = nullptr;
    double* d_h_pol = nullptr;
    double* d_d_a_pol = nullptr;
    double* d_d_h_pol = nullptr;
    double* d_adjust = nullptr;
    
    // Grid on device
    double* d_m_grid = nullptr;
    double* d_a_grid = nullptr;
    double* d_h_grid = nullptr;
    double* d_z_grid = nullptr;
    double* d_Pi = nullptr;
    
    // Expectations
    double* d_E_V = nullptr;
    double* d_E_Vm = nullptr;
    
    // Params (flattened)
    double beta, r_m, r_a, r_h;
    double chi0, chi1, chi2, sigma;
    double tax_lambda, tax_tau, tax_transfer, tax_cgt;
    double wealth_tax_rate, wealth_tax_thresh;
    
public:
    // Construct with raw sizes
    GpuBackend(int Nm, int Na, int Nh, int Nz);
    ~GpuBackend();

    // Upload grids and parameters
    void upload_grids(const double* m_grid, const double* a_grid, const double* h_grid, const double* z_grid, 
                      const double* Pi, int Nm, int Na, int Nh, int Nz);
    void set_params(double beta, double r_m, double r_a, double r_h, 
                    double chi0, double chi1, double chi2, double sigma,
                    double tax_lambda, double tax_tau, double tax_transfer, double tax_cgt,
                    double wealth_tax_rate, double wealth_tax_thresh);
    
    // Core solver - takes flat arrays, returns diff
    double solve_bellman_iteration(
        const double* h_V_in, const double* h_c_in,
        double* h_V_out, double* h_c_out, double* h_m_out, double* h_a_out, double* h_h_out, 
        double* h_d_a_out, double* h_d_h_out, double* h_adjust_out,
        int total_size);
        
    // --- Phase G9: Monte Carlo Simulation ---
    // Simulate N_sim agents for T_periods
    // Returns aggregate stats (e.g., mean wealth per period) if needed, 
    // or just fills h_sim_results (N_sim * 3 [m,a,h] final state?)
    // For now, let's just print stats and return nothing, or return mean wealth history.
    void simulate(int N_sim, int T_periods, std::vector<double>& mean_wealth_history);

private:
    void allocate_memory(int total);
    void free_memory();
    
    // Simulation
    void* d_rng_states = nullptr; // curandState* cast to void* to avoid exposing CUDA headers
};

} // namespace Monad
