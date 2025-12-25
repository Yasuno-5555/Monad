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
    
    int low = 1;
    int high = n - 1;
    int idx = high;
    
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (x[mid] >= xi) {
            idx = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    
    double t = (xi - x[idx-1]) / (x[idx] - x[idx-1]);
    return y[idx-1] + t * (y[idx] - y[idx-1]);
}

// Bilinear interpolation for E_V / E_Vm
// Uses lower_bound semantics matching CPU exactly
// Trilinear/3D interpolation for E_V / E_Vm
// (iz, m, a, h) -> value
__device__ double d_interp_E(const double* E_data, int iz, double m, double a, double h,
                             const double* m_grid, int Nm,
                             const double* a_grid, int Na,
                             const double* h_grid, int Nh, int Nz) {
                             
    auto interp_2d = [&](int ih) -> double {
        auto interp_m_at_ia = [&](int ia) -> double {
            int base = iz * Nh * Na * Nm + ih * Na * Nm + ia * Nm;
            double m0 = m_grid[0], m_end = m_grid[Nm-1];
            if (m <= m0) return __ldg(&E_data[base]);
            if (m >= m_end) return __ldg(&E_data[base + Nm - 1]);
            
            int low = 1, high = Nm - 1, im = high;
            while (low <= high) {
                int mid = low + (high - low) / 2;
                if (m_grid[mid] >= m) { im = mid; high = mid - 1; }
                else { low = mid + 1; }
            }
            double m_lo = m_grid[im-1], m_hi = m_grid[im];
            double t = (m - m_lo) / (m_hi - m_lo);
            return __ldg(&E_data[base + im - 1]) + t * (__ldg(&E_data[base + im]) - __ldg(&E_data[base + im - 1]));
        };

        double a0 = a_grid[0], a_end = a_grid[Na-1];
        if (a <= a0) return interp_m_at_ia(0);
        if (a >= a_end) return interp_m_at_ia(Na-1);
        
        int low = 1, high = Na - 1, ia = high;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (a_grid[mid] >= a) { ia = mid; high = mid - 1; }
            else { low = mid + 1; }
        }
        double a_lo = a_grid[ia-1], a_hi = a_grid[ia];
        double t_a = (a - a_lo) / (a_hi - a_lo);
        return interp_m_at_ia(ia - 1) + t_a * (interp_m_at_ia(ia) - interp_m_at_ia(ia - 1));
    };

    double h0 = h_grid[0], h_end = h_grid[Nh-1];
    if (h <= h0) return interp_2d(0);
    if (h >= h_end) return interp_2d(Nh-1);
    
    int low = 1, high = Nh - 1, ih = high;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (h_grid[mid] >= h) { ih = mid; high = mid - 1; }
        else { low = mid + 1; }
    }
    double h_lo = h_grid[ih-1], h_hi = h_grid[ih];
    double t_h = (h - h_lo) / (h_hi - h_lo);
    return interp_2d(ih - 1) + t_h * (interp_2d(ih) - interp_2d(ih - 1));
}

// ============================================
// Expectations Kernel
// ============================================
__global__ void expectations_kernel(
    const double* __restrict__ V_in, const double* __restrict__ c_in, const double* __restrict__ Pi,
    double* __restrict__ E_V, double* __restrict__ E_Vm,
    double sigma, double r_m, int Nm, int Na, int Nh, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nh * Nz;
    if (idx >= total) return;
    
    int iz = idx / (Nh * Na * Nm);
    int rem = idx % (Nh * Na * Nm);
    int ih = rem / (Na * Nm);
    rem %= (Na * Nm);
    int ia = rem / Nm;
    int im = rem % Nm;
    
    double ev = 0.0, evm = 0.0;
    for (int jz = 0; jz < Nz; ++jz) {
        double prob = __ldg(&Pi[iz * Nz + jz]);
        if (prob > 1e-12) {
            int j_idx = jz * Nh * Na * Nm + ih * Na * Nm + ia * Nm + im;
            ev += prob * __ldg(&V_in[j_idx]);
            double c_next = fmax(__ldg(&c_in[j_idx]), 1e-9);
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
    const double* __restrict__ m_grid, const double* __restrict__ a_grid, const double* __restrict__ h_grid, const double* __restrict__ z_grid,
    const double* __restrict__ E_V, const double* __restrict__ E_Vm,
    double* __restrict__ V_out, double* __restrict__ c_out, double* __restrict__ m_out, double* __restrict__ a_out, double* __restrict__ h_out,
    double beta, double r_m, double r_a, double r_h, double chi, double sigma,
    double tax_lambda, double tax_tau, double tax_transfer, double tax_cgt,
    int Nm, int Na, int Nh, int Nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nh * Nz;
    if (idx >= total) return;
    
    int iz = idx / (Nh * Na * Nm);
    int rem = idx % (Nh * Na * Nm);
    int ih = rem / (Na * Nm);
    rem %= (Na * Nm);
    int ia = rem / Nm;
    int im = rem % Nm;

    // Cache grids
    __shared__ double s_m_grid[128];
    __shared__ double s_a_grid[128];
    __shared__ double s_h_grid[128];
    for (int i = threadIdx.x; i < Nm && i < 128; i += blockDim.x) s_m_grid[i] = m_grid[i];
    for (int i = threadIdx.x; i < Na && i < 128; i += blockDim.x) s_a_grid[i] = a_grid[i];
    for (int i = threadIdx.x; i < Nh && i < 128; i += blockDim.x) s_h_grid[i] = h_grid[i];
    __syncthreads();
    
    double z_val = __ldg(&z_grid[iz]);
    double net_income = (z_val <= 1e-9) ? tax_transfer : (tax_lambda * pow(z_val, 1.0 - tax_tau) + tax_transfer);
    
    double a_curr = s_a_grid[ia];
    double h_curr = s_h_grid[ih];
    double m_fixed = s_m_grid[im];

    // Net returns
    double r_a_net = r_a * (1.0 - tax_cgt);
    double r_h_net = r_h * (1.0 - tax_cgt);
    
    // ============================================
    // No Adjustment
    // ============================================
    double a_next_no = a_curr * (1.0 + r_a_net);
    double h_next_no = h_curr * (1.0 + r_h_net);
    
    const int MAX_N = 128;
    double c_endo[MAX_N], m_endo[MAX_N];
    
    for (int j = 0; j < Nm && j < MAX_N; ++j) {
        double m_next_j = s_m_grid[j];
        double emv = d_interp_E(E_Vm, iz, m_next_j, a_next_no, h_next_no, s_m_grid, Nm, s_a_grid, Na, s_h_grid, Nh, Nz);
        double c = d_inv_u_prime(beta * emv, sigma);
        m_endo[j] = (c + m_next_j - net_income) / (1.0 + r_m);
        c_endo[j] = c;
    }
    
    double c_no, m_prime_no;
    if (m_fixed < m_endo[0]) {
        m_prime_no = s_m_grid[0];
        c_no = (1.0 + r_m) * m_fixed + net_income - m_prime_no;
    } else {
        c_no = d_interp_1d(m_endo, Nm, c_endo, m_fixed);
        m_prime_no = d_interp_1d(m_endo, Nm, s_m_grid, m_fixed);
    }
    if (c_no < 1e-9) c_no = 1e-9;
    
    double ev_no = d_interp_E(E_V, iz, m_prime_no, a_next_no, h_next_no, s_m_grid, Nm, s_a_grid, Na, s_h_grid, Nh, Nz);
    double best_val = d_u(c_no, sigma) + beta * ev_no;
    double best_c = c_no;
    double best_m_prime = m_prime_no;
    double best_a_next = a_next_no;
    double best_h_next = h_next_no;
    double best_adjust = 0.0;
    
    // ============================================
    // Adjustment (A only for now)
    // ============================================
    for (int ia_next = 0; ia_next < Na; ++ia_next) {
        double a_next_adj = s_a_grid[ia_next];
        double d_a = a_next_adj - a_curr * (1.0 + r_a_net);
        double total_outflow = d_a + (fabs(d_a) < 1e-9 ? 0.0 : chi * d_a * d_a);

        for (int j = 0; j < Nm && j < MAX_N; ++j) {
            double m_next_j = s_m_grid[j];
            double emv = d_interp_E(E_Vm, iz, m_next_j, a_next_adj, h_next_no, s_m_grid, Nm, s_a_grid, Na, s_h_grid, Nh, Nz);
            double c = d_inv_u_prime(beta * emv, sigma);
            m_endo[j] = (c + m_next_j - net_income + total_outflow) / (1.0 + r_m);
            c_endo[j] = c;
        }
        
        double c_adj, m_prime_adj;
        if (m_fixed < m_endo[0]) {
            m_prime_adj = s_m_grid[0];
            c_adj = (1.0 + r_m) * m_fixed + net_income - total_outflow - m_prime_adj;
        } else {
            c_adj = d_interp_1d(m_endo, Nm, c_endo, m_fixed);
            m_prime_adj = d_interp_1d(m_endo, Nm, s_m_grid, m_fixed);
        }
        
        if (c_adj > 1e-9) {
            double ev_adj = d_interp_E(E_V, iz, m_prime_adj, a_next_adj, h_next_no, s_m_grid, Nm, s_a_grid, Na, s_h_grid, Nh, Nz);
            double val_adj = d_u(c_adj, sigma) + beta * ev_adj;
            if (val_adj > best_val) {
                best_val = val_adj;
                best_c = c_adj;
                best_m_prime = m_prime_adj;
                best_a_next = a_next_adj;
                best_h_next = h_next_no;
                best_adjust = 1.0;
            }
        }
    }
    
    V_out[idx] = best_val;
    c_out[idx] = best_c;
    m_out[idx] = best_m_prime;
    a_out[idx] = best_a_next;
    h_out[idx] = best_h_next;
}

// ============================================
// Host Implementation
// ============================================

GpuBackend::GpuBackend(int Nm, int Na, int Nh, int Nz) {
    grid_sizes.N_m = Nm;
    grid_sizes.N_a = Na;
    grid_sizes.N_h = Nh;
    grid_sizes.N_z = Nz;
    grid_sizes.total_size = Nm * Na * Nh * Nz;
    
    std::cerr << "[GpuBackend] Initializing with Total States: " << grid_sizes.total_size << std::endl;
    try {
        allocate_memory(grid_sizes.total_size);
    } catch (const std::exception& e) {
        std::cerr << "[GpuBackend] Fatal Error during allocation: " << e.what() << std::endl;
        throw;
    }
}

GpuBackend::~GpuBackend() { free_memory(); }

void GpuBackend::allocate_memory(int total) {
    size_t sz = total * sizeof(double);
    std::cerr << "[GpuBackend] Allocating GPU memory. Double array size: " << sz / (1024.0*1024.0) << " MB" << std::endl;
    
    auto check_malloc = [&](void** ptr, size_t s, const char* name) {
        cudaError_t err = cudaMalloc(ptr, s);
        if (err != cudaSuccess) {
            std::cerr << "[GpuBackend] Failed to allocate " << name << " (" << s << " bytes): " 
                      << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(std::string("CUDA Malloc failed for ") + name);
        }
    };

    check_malloc((void**)&d_V, sz, "d_V");
    check_malloc((void**)&d_V_next, sz, "d_V_next");
    check_malloc((void**)&d_c, sz, "d_c");
    check_malloc((void**)&d_c_out, sz, "d_c_out");  
    check_malloc((void**)&d_m_pol, sz, "d_m_pol");
    check_malloc((void**)&d_a_pol, sz, "d_a_pol");
    check_malloc((void**)&d_h_pol, sz, "d_h_pol");
    check_malloc((void**)&d_d_a_pol, sz, "d_d_a_pol");
    check_malloc((void**)&d_d_h_pol, sz, "d_d_h_pol");
    check_malloc((void**)&d_adjust, sz, "d_adjust");
    check_malloc((void**)&d_E_V, sz, "d_E_V");
    check_malloc((void**)&d_E_Vm, sz, "d_E_Vm");
    
    check_malloc((void**)&d_m_grid, grid_sizes.N_m * sizeof(double), "d_m_grid");
    check_malloc((void**)&d_a_grid, grid_sizes.N_a * sizeof(double), "d_a_grid");
    check_malloc((void**)&d_h_grid, grid_sizes.N_h * sizeof(double), "d_h_grid");
    check_malloc((void**)&d_z_grid, grid_sizes.N_z * sizeof(double), "d_z_grid");
    check_malloc((void**)&d_Pi, grid_sizes.N_z * grid_sizes.N_z * sizeof(double), "d_Pi");
    
    std::cerr << "[GpuBackend] Memory allocation successful." << std::endl;
}

void GpuBackend::free_memory() {
    if (d_V) cudaFree(d_V);
    if (d_V_next) cudaFree(d_V_next);
    if (d_c) cudaFree(d_c);
    if (d_c_out) cudaFree(d_c_out);
    if (d_m_pol) cudaFree(d_m_pol);
    if (d_a_pol) cudaFree(d_a_pol);
    if (d_h_pol) cudaFree(d_h_pol);
    if (d_d_a_pol) cudaFree(d_d_a_pol);
    if (d_d_h_pol) cudaFree(d_d_h_pol);
    if (d_adjust) cudaFree(d_adjust);
    if (d_E_V) cudaFree(d_E_V);
    if (d_E_Vm) cudaFree(d_E_Vm);
    if (d_m_grid) cudaFree(d_m_grid);
    if (d_a_grid) cudaFree(d_a_grid);
    if (d_h_grid) cudaFree(d_h_grid);
    if (d_z_grid) cudaFree(d_z_grid);
    if (d_Pi) cudaFree(d_Pi);
}

void GpuBackend::upload_grids(const double* m, const double* a, const double* h, const double* z,
                               const double* Pi, int Nm, int Na, int Nh, int Nz) {
    CUDA_CHECK(cudaMemcpy(d_m_grid, m, Nm * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_grid, a, Na * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_grid, h, Nh * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_grid, z, Nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi, Pi, Nz * Nz * sizeof(double), cudaMemcpyHostToDevice));
}

void GpuBackend::set_params(double b, double rm, double ra, double rh, double c, double s,
                            double tl, double tt, double ttr, double tcgt) {
    beta = b; r_m = rm; r_a = ra; r_h = rh; chi = c; sigma = s;
    tax_lambda = tl; tax_tau = tt; tax_transfer = ttr; tax_cgt = tcgt;
}

double GpuBackend::solve_bellman_iteration(
    const double* h_V_in, const double* h_c_in,
    double* h_V_out, double* h_c_out, double* h_m_out, double* h_a_out, double* h_h_out,
    double* h_d_a_out, double* h_d_h_out, double* h_adjust_out, int total_size)
{
    size_t sz = total_size * sizeof(double);
    int Nm = grid_sizes.N_m, Na = grid_sizes.N_a, Nh = grid_sizes.N_h, Nz = grid_sizes.N_z;
    
    CUDA_CHECK(cudaMemcpy(d_V, h_V_in, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c_in, sz, cudaMemcpyHostToDevice));
    
    int threads = 128;
    int blocks = (total_size + threads - 1) / threads;
    
    expectations_kernel<<<blocks, threads>>>(
        d_V, d_c, d_Pi, d_E_V, d_E_Vm, sigma, r_m, Nm, Na, Nh, Nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    bellman_kernel<<<blocks, threads>>>(
        d_m_grid, d_a_grid, d_h_grid, d_z_grid, d_E_V, d_E_Vm,
        d_V_next, d_c_out, d_m_pol, d_a_pol, d_h_pol,
        beta, r_m, r_a, r_h, chi, sigma, tax_lambda, tax_tau, tax_transfer, tax_cgt, Nm, Na, Nh, Nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_V_out, d_V_next, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_out, d_c_out, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m_out, d_m_pol, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_a_out, d_a_pol, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h_out, d_h_pol, sz, cudaMemcpyDeviceToHost));
    
    return 0.0;
}

} // namespace Monad

#else

namespace Monad {
    GpuBackend::GpuBackend(int, int, int, int) {}
    GpuBackend::~GpuBackend() {}
    void GpuBackend::allocate_memory(int) {}
    void GpuBackend::free_memory() {}
    void GpuBackend::upload_grids(const double*, const double*, const double*, const double*, const double*, int, int, int, int) {}
    void GpuBackend::set_params(double, double, double, double, double, double, double, double, double, double) {}
    double GpuBackend::solve_bellman_iteration(const double*, const double*, double*, double*, double*, double*, double*, double*, double*, double*, int) { return 0.0; }
}

#endif
