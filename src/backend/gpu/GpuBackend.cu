#include "GpuBackend.cuh"
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>

#ifdef MONAD_GPU
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
    } \
} while(0)

namespace Monad {

// ============================================
// SMOKE TEST KERNEL (always keep this)
// ============================================
extern "C" __global__ void smoke_test_kernel(double* V_out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        V_out[0] = 777.0;
    }
}

// ============================================
// Device Helper Functions
// ============================================

__device__ double d_u(double c, double sigma) {
    if (c <= 1e-9) return -1e9;
    if (fabs(sigma - 1.0) < 1e-5) return log(c);
    return pow(c, 1.0 - sigma) / (1.0 - sigma);
}

__device__ double d_inv_u_prime(double val, double sigma) {
    if (val <= 1e-9) return 1e9;
    return pow(val, -1.0 / sigma);
}

// Simple 1D interpolation with lower_bound semantics (matching CPU)
__device__ double d_interp_1d(const double* x, int n, const double* y, double xi) {
    if (xi <= x[0]) return y[0];
    if (xi >= x[n-1]) return y[n-1];
    
    // lower_bound: find first where x[idx] >= xi
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] >= xi) { idx = i; break; }
    }
    // idx is now where x[idx] >= xi
    // Interpolate between idx-1 and idx
    double t = (xi - x[idx-1]) / (x[idx] - x[idx-1]);
    return y[idx-1] + t * (y[idx] - y[idx-1]);
}

// Bilinear interpolation for E_V / E_Vm
// Uses lower_bound semantics matching CPU exactly
__device__ double d_interp_E(const double* E_data, int iz, double m, double a,
                             const double* m_grid, int Nm,
                             const double* a_grid, int Na, int Nz) {
    // Interpolate along m for a given ia
    auto interp_m_at_ia = [&](int ia) -> double {
        int base = iz * Na * Nm + ia * Nm;
        if (m <= m_grid[0]) return E_data[base];
        if (m >= m_grid[Nm-1]) return E_data[base + Nm - 1];
        
        // lower_bound: find first where m_grid[im] >= m
        int im = 0;
        for (int i = 0; i < Nm; ++i) {
            if (m_grid[i] >= m) { im = i; break; }
        }
        // im is now the index where m_grid[im] >= m
        // Interpolate between im-1 and im
        double t = (m - m_grid[im-1]) / (m_grid[im] - m_grid[im-1]);
        return E_data[base + im - 1] + t * (E_data[base + im] - E_data[base + im - 1]);
    };
    
    // Interpolate along a
    if (a <= a_grid[0]) return interp_m_at_ia(0);
    if (a >= a_grid[Na-1]) return interp_m_at_ia(Na-1);
    
    // lower_bound: find first where a_grid[ia] >= a
    int ia = 0;
    for (int i = 0; i < Na; ++i) {
        if (a_grid[i] >= a) { ia = i; break; }
    }
    
    double t_a = (a - a_grid[ia-1]) / (a_grid[ia] - a_grid[ia-1]);
    double v_lo = interp_m_at_ia(ia - 1);
    double v_hi = interp_m_at_ia(ia);
    return v_lo + t_a * (v_hi - v_lo);
}

// ============================================
// Expectations Kernel
// ============================================
__global__ void expectations_kernel(
    const double* V_in, const double* c_in, const double* Pi,
    double* E_V, double* E_Vm,
    double sigma, double r_m, int Nm, int Na, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nz;
    if (idx >= total) return;
    
    int iz = idx / (Na * Nm);
    int rem = idx % (Na * Nm);
    int ia = rem / Nm;
    int im = rem % Nm;
    
    double ev = 0.0, evm = 0.0;
    for (int jz = 0; jz < Nz; ++jz) {
        double prob = Pi[iz * Nz + jz];
        if (prob > 1e-12) {
            int j_idx = jz * Na * Nm + ia * Nm + im;
            ev += prob * V_in[j_idx];
            double c_next = fmax(c_in[j_idx], 1e-9);
            evm += prob * pow(c_next, -sigma) * (1.0 + r_m);
        }
    }
    E_V[idx] = ev;
    E_Vm[idx] = evm;
}

// ============================================
// Bellman Kernel (No Adjustment - Step 1)
// ============================================
__global__ void bellman_kernel(
    const double* m_grid, const double* a_grid, const double* z_grid,
    const double* E_V, const double* E_Vm,
    double* V_out, double* c_out, double* m_out, double* a_out,
    double beta, double r_m, double r_a, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer,
    int Nm, int Na, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nz;
    if (idx >= total) return;
    
    int iz = idx / (Na * Nm);
    int rem = idx % (Na * Nm);
    int ia = rem / Nm;
    int im = rem % Nm;
    
    double z_val = z_grid[iz];
    // after_tax matching CPU: if z_val <= 1e-9, return only transfer
    double net_income;
    if (z_val <= 1e-9) {
        net_income = tax_transfer;
    } else {
        net_income = tax_lambda * pow(z_val, 1.0 - tax_tau) + tax_transfer;
    }
    double a_curr = a_grid[ia];
    double m_fixed = m_grid[im];
    
    // ============================================
    // No Adjustment
    // ============================================
    double a_next_no = a_curr * (1.0 + r_a);
    
    const int MAX_N = 200;
    double c_endo[MAX_N], m_endo[MAX_N];
    
    // Build EGM for No Adjustment
    for (int j = 0; j < Nm && j < MAX_N; ++j) {
        double m_next_j = m_grid[j];
        double emv = d_interp_E(E_Vm, iz, m_next_j, a_next_no, m_grid, Nm, a_grid, Na, Nz);
        double c = d_inv_u_prime(beta * emv, sigma);
        m_endo[j] = (c + m_next_j - net_income) / (1.0 + r_m);
        c_endo[j] = c;
    }
    
    double c_no, m_prime_no;
    if (m_fixed < m_endo[0]) {
        m_prime_no = m_grid[0];
        c_no = (1.0 + r_m) * m_fixed + net_income - m_prime_no;
    } else {
        c_no = d_interp_1d(m_endo, Nm, c_endo, m_fixed);
        m_prime_no = d_interp_1d(m_endo, Nm, m_grid, m_fixed);
    }
    if (c_no < 1e-9) c_no = 1e-9;
    
    double ev_no = d_interp_E(E_V, iz, m_prime_no, a_next_no, m_grid, Nm, a_grid, Na, Nz);
    double val_no = d_u(c_no, sigma) + beta * ev_no;
    
    // Initialize with No Adjustment values
    double best_val = val_no;
    double best_c = c_no;
    double best_m_prime = m_prime_no;
    double best_a_next = a_next_no;
    double best_d = 0.0;
    double best_adjust = 0.0;
    
    // ============================================
    // Adjustment: try all ia_next
    // ============================================
    for (int ia_next = 0; ia_next < Na; ++ia_next) {
        double a_next_adj = a_grid[ia_next];
        double d = a_next_adj - a_curr * (1.0 + r_a);
        
        // Adjustment cost: chi * d^2
        double cost = (fabs(d) < 1e-9) ? 0.0 : (chi * d * d);
        double total_outflow = d + cost;
        
        // Build EGM for this adjustment target
        // Using interp_1d_slice_m equivalent: d_interp_E at fixed a_next_adj
        for (int j = 0; j < Nm && j < MAX_N; ++j) {
            double m_next_j = m_grid[j];
            // E_Vm at (iz, m_next_j, a_next_adj)
            double emv = d_interp_E(E_Vm, iz, m_next_j, a_next_adj, m_grid, Nm, a_grid, Na, Nz);
            double c = d_inv_u_prime(beta * emv, sigma);
            m_endo[j] = (c + m_next_j - net_income + total_outflow) / (1.0 + r_m);
            c_endo[j] = c;
        }
        
        // Solve for this m_fixed
        double c_adj, m_prime_adj;
        if (m_fixed < m_endo[0]) {
            m_prime_adj = m_grid[0];
            c_adj = (1.0 + r_m) * m_fixed + net_income - total_outflow - m_prime_adj;
        } else {
            c_adj = d_interp_1d(m_endo, Nm, c_endo, m_fixed);
            m_prime_adj = d_interp_1d(m_endo, Nm, m_grid, m_fixed);
        }
        
        if (c_adj > 1e-9) {
            double ev_adj = d_interp_E(E_V, iz, m_prime_adj, a_next_adj, m_grid, Nm, a_grid, Na, Nz);
            double val_adj = d_u(c_adj, sigma) + beta * ev_adj;
            
            if (val_adj > best_val) {
                best_val = val_adj;
                best_c = c_adj;
                best_m_prime = m_prime_adj;
                best_a_next = a_next_adj;
                best_d = d;
                best_adjust = 1.0;
            }
        }
    }
    
    // Write best result
    V_out[idx] = best_val;
    c_out[idx] = best_c;
    m_out[idx] = best_m_prime;
    a_out[idx] = best_a_next;
}

// ============================================
// Host Implementation
// ============================================

GpuBackend::GpuBackend(int Nm, int Na, int Nz) {
    grid_sizes.N_m = Nm;
    grid_sizes.N_a = Na;
    grid_sizes.N_z = Nz;
    grid_sizes.total_size = Nm * Na * Nz;
    allocate_memory(grid_sizes.total_size);
}

GpuBackend::~GpuBackend() { free_memory(); }

void GpuBackend::allocate_memory(int total) {
    size_t sz = total * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_V, sz));
    CUDA_CHECK(cudaMalloc(&d_V_next, sz));
    CUDA_CHECK(cudaMalloc(&d_c, sz));
    CUDA_CHECK(cudaMalloc(&d_c_out, sz));  // Separate output buffer
    CUDA_CHECK(cudaMalloc(&d_m_pol, sz));
    CUDA_CHECK(cudaMalloc(&d_a_pol, sz));
    CUDA_CHECK(cudaMalloc(&d_d_pol, sz));
    CUDA_CHECK(cudaMalloc(&d_adjust, sz));
    CUDA_CHECK(cudaMalloc(&d_E_V, sz));
    CUDA_CHECK(cudaMalloc(&d_E_Vm, sz));
    CUDA_CHECK(cudaMalloc(&d_m_grid, grid_sizes.N_m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a_grid, grid_sizes.N_a * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z_grid, grid_sizes.N_z * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Pi, grid_sizes.N_z * grid_sizes.N_z * sizeof(double)));
}

void GpuBackend::free_memory() {
    if (d_V) cudaFree(d_V);
    if (d_V_next) cudaFree(d_V_next);
    if (d_c) cudaFree(d_c);
    if (d_c_out) cudaFree(d_c_out);
    if (d_m_pol) cudaFree(d_m_pol);
    if (d_a_pol) cudaFree(d_a_pol);
    if (d_d_pol) cudaFree(d_d_pol);
    if (d_adjust) cudaFree(d_adjust);
    if (d_E_V) cudaFree(d_E_V);
    if (d_E_Vm) cudaFree(d_E_Vm);
    if (d_m_grid) cudaFree(d_m_grid);
    if (d_a_grid) cudaFree(d_a_grid);
    if (d_z_grid) cudaFree(d_z_grid);
    if (d_Pi) cudaFree(d_Pi);
}

void GpuBackend::upload_grids(const double* m, const double* a, const double* z,
                              const double* Pi, int Nm, int Na, int Nz) {
    CUDA_CHECK(cudaMemcpy(d_m_grid, m, Nm * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_grid, a, Na * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_grid, z, Nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi, Nz * Nz * sizeof(double), cudaMemcpyHostToDevice));
}

void GpuBackend::set_params(double b, double rm, double ra, double c, double s,
                            double tl, double tt, double ttr) {
    beta = b; r_m = rm; r_a = ra; chi = c; sigma = s;
    tax_lambda = tl; tax_tau = tt; tax_transfer = ttr;
}

double GpuBackend::solve_bellman_iteration(
    const double* h_V_in, const double* h_c_in,
    double* h_V_out, double* h_c_out, double* h_m_out, double* h_a_out,
    double* h_d_out, double* h_adjust_out, int total_size)
{
    size_t sz = total_size * sizeof(double);
    int Nm = grid_sizes.N_m, Na = grid_sizes.N_a, Nz = grid_sizes.N_z;
    
    CUDA_CHECK(cudaMemcpy(d_V, h_V_in, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c_in, sz, cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    // Expectations
    expectations_kernel<<<blocks, threads>>>(
        d_V, d_c, d_Pi, d_E_V, d_E_Vm, sigma, r_m, Nm, Na, Nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Bellman - writes to d_V_next and d_c_out (not d_c)
    bellman_kernel<<<blocks, threads>>>(
        d_m_grid, d_a_grid, d_z_grid, d_E_V, d_E_Vm,
        d_V_next, d_c_out, d_m_pol, d_a_pol,
        beta, r_m, r_a, chi, sigma, tax_lambda, tax_tau, tax_transfer, Nm, Na, Nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_V_out, d_V_next, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_out, d_c_out, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m_out, d_m_pol, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_a_out, d_a_pol, sz, cudaMemcpyDeviceToHost));
    
    std::fill(h_d_out, h_d_out + total_size, 0.0);
    std::fill(h_adjust_out, h_adjust_out + total_size, 0.0);
    
    return 0.0;
}

} // namespace Monad

#else

namespace Monad {
    GpuBackend::GpuBackend(int, int, int) {}
    GpuBackend::~GpuBackend() {}
    void GpuBackend::allocate_memory(int) {}
    void GpuBackend::free_memory() {}
    void GpuBackend::upload_grids(const double*, const double*, const double*, const double*, int, int, int) {}
    void GpuBackend::set_params(double, double, double, double, double, double, double, double) {}
    double GpuBackend::solve_bellman_iteration(const double*, const double*, double*, double*, double*, double*, double*, double*, int) { return 0.0; }
}

#endif
