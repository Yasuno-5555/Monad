#pragma once

#ifdef MONAD_GPU
#include <cuda_runtime.h>
#endif

namespace Monad {

// Forward declare types needed by GPU backend
// These avoid pulling in Eigen or complex headers
struct GpuGridSizes {
    int N_m;
    int N_a;
    int N_z;
    int total_size;
};

class GpuBackend {
    GpuGridSizes grid_sizes;
    
    // Device pointers
    double* d_V = nullptr;
    double* d_V_next = nullptr;
    double* d_c = nullptr;
    double* d_c_out = nullptr;  // Separate output for c_pol
    double* d_m_pol = nullptr;
    double* d_a_pol = nullptr;
    double* d_d_pol = nullptr;
    double* d_adjust = nullptr;
    
    // Grid on device
    double* d_m_grid = nullptr;
    double* d_a_grid = nullptr;
    double* d_z_grid = nullptr;
    double* d_Pi = nullptr;
    
    // Expectations
    double* d_E_V = nullptr;
    double* d_E_Vm = nullptr;
    
    // Params (flattened)
    double beta, r_m, r_a, chi, sigma;
    double tax_lambda, tax_tau, tax_transfer;
    
public:
    // Construct with raw sizes
    GpuBackend(int Nm, int Na, int Nz);
    ~GpuBackend();

    // Upload grids and parameters
    void upload_grids(const double* m_grid, const double* a_grid, const double* z_grid, 
                      const double* Pi, int Nm, int Na, int Nz);
    void set_params(double beta, double r_m, double r_a, double chi, double sigma,
                    double tax_lambda, double tax_tau, double tax_transfer);
    
    // Core solver - takes flat arrays, returns diff
    double solve_bellman_iteration(
        const double* h_V_in, const double* h_c_in,
        double* h_V_out, double* h_c_out, double* h_m_out, double* h_a_out, 
        double* h_d_out, double* h_adjust_out,
        int total_size);
    
private:
    void allocate_memory(int total);
    void free_memory();
};

} // namespace Monad
