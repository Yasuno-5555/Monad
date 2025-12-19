#include "CudaUtils.hpp"
#include "../Dual.hpp"
#include <cmath>
#include <cfloat>

namespace Monad {

// --- Device Helpers ---

template <typename T>
__device__ T d_u(T c, double sigma) {
    // Note: c might be Dual. Comparisons work.
    if (c <= 1e-9) return -1e9;
    // fabs for Dual? Dual.hpp needs to support generic fabs or overload.
    // Actually Dual.hpp defines abs(). We should use abs() for T?
    // But sigma is double.
    if (fabs(sigma - 1.0) < 1e-5) return log(c);
    return pow(c, 1.0 - sigma) / (1.0 - sigma);
}

template <typename T>
__device__ T d_u_prime(T c, double sigma) {
    return pow(c, -sigma);
}

template <typename T>
__device__ T d_inv_u_prime(T val, double sigma) {
    if (val <= 1e-9) return 1e9; 
    return pow(val, -1.0/sigma);
}

// Linear Interpolation on device (Templated for Dual xi)
// Linear Interpolation (Fully Templated for Data and Query)
template <typename DataT, typename QueryT>
__device__ QueryT d_interp_1d(const DataT* x, const DataT* y, int n, QueryT xi) {
    if (xi <= x[0]) return y[0];
    if (xi >= x[n-1]) return y[n-1];
    
    // Binary Search
    int left = 0;
    int right = n - 1;
    
    while (left < right - 1) { 
        int mid = (left + right) / 2;
        if (x[mid] <= xi) left = mid;
        else right = mid;
    }
    
    // Interpolation fraction t must retain derivative info if xi is Dual
    // But if DataT is Dual and QueryT is Dual, we need generic arithmetic.
    // Assuming +/-/*// are overloaded.
    
    QueryT t = (xi - x[left]) / (x[right] - x[left]);
    return y[left] + t * (y[right] - y[left]);
}

// 2D slice interpolation
template <typename T>
__device__ T d_interp_2d(const double* data_slice, 
                              const double* m_grid, int Nm, 
                              const double* a_grid, int Na, 
                              T m, T a) {
    // 1. Find 'a' weights
    int ia = 0;
    if (a <= a_grid[0]) ia = 0;
    else if (a >= a_grid[Na-1]) ia = Na - 2;
    else {
        int l=0, r=Na-1;
        while(l < r-1) {
            int mid = (l+r)/2;
            if(a_grid[mid] <= a) l=mid; else r=mid;
        }
        ia = l;
    }
    T ta = (a - a_grid[ia]) / (a_grid[ia+1] - a_grid[ia]);
    
    // 2. Interpolate 'm' at ia and ia+1
    // We pass addresses of double arrays.
    T v_low  = d_interp_1d(m_grid, &data_slice[ia*Nm], Nm, m);
    T v_high = d_interp_1d(m_grid, &data_slice[(ia+1)*Nm], Nm, m);
    
    return v_low + ta * (v_high - v_low);
}

// --- Kernels ---

// Kernel 1: Expectations (Tensor Contraction)
// E[V](m, a, z) = sum_{z'} Pi(z, z') * V(m, a, z')
// Each thread computes one (im, ia, iz) point.
__global__ void compute_expectations_kernel(
    double* d_EV, double* d_EVm, // Outputs
    const double* d_V, const double* d_c, const double* d_m_grid, 
    const double* d_Pi, // Transition Matrix (flat z*z)
    int Nm, int Na, int Nz, 
    double r_m, double sigma) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nz;
    if (idx >= total) return;
    
    // Decode index -> (im, ia, iz)
    int im = idx % Nm;
    int ia = (idx / Nm) % Na;
    int iz = (idx / (Nm * Na)); // Current z
    
    double ev = 0.0;
    double evm = 0.0;
    
    // Loop over next z
    for (int next_iz = 0; next_iz < Nz; ++next_iz) {
        double prob = d_Pi[iz * Nz + next_iz];
        if (prob > 1e-10) {
            int next_idx = (next_iz * Na * Nm) + (ia * Nm) + im; // V(m, a, z')
            
            ev += prob * d_V[next_idx];
            
            // Marginal Value: u'(c(z')) * (1+r)
            double c_next = d_c[next_idx];
            double u_p = d_u_prime(c_next, sigma);
            evm += prob * u_p * (1.0 + r_m);
        }
    }
    
    d_EV[idx] = ev;
    d_EVm[idx] = evm;
}

// Wrapper to launch Expectation Kernel
void launch_expectations(CudaBackend& backend, double r_m, double sigma) {
    int total = backend.N_m * backend.N_a * backend.N_z;
    int threads_per_block = 256;
    int blocks = (total + threads_per_block - 1) / threads_per_block;
    
    // Launch
    compute_expectations_kernel<<<blocks, threads_per_block>>>(
        backend.d_EV, 
        backend.d_EVm,
        backend.d_V_next, // Note: We use V_next (from previous iteration guess) as input
        backend.d_c_pol,
        backend.d_m_grid,
        backend.d_Pi,
        backend.N_m, backend.N_a, backend.N_z,
        r_m, sigma
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In real app, log error or throw
        // printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// Kernel 2: Bellman (No Adjustment)
// Parallelism: One Block per (ia, iz) slice. Threads per block = Nm.
__global__ void bellman_no_adjust_kernel(
    double* d_V, double* d_c, double* d_m_pol, double* d_a_pol, double* d_d_pol, double* d_flag,
    const double* d_EV, const double* d_EVm,
    const double* d_m_grid, const double* d_a_grid,
    int Nm, int Na, int Nz,
    double beta, double r_m, double r_a, double sigma, double m_min,
    double z_val_fixed_for_now_or_need_grid, 
    const double* d_z_grid, const double* d_tax_params // Simplifying: pass net income? 
    // Ideally we pass full z grid and compute net income here, 
    // OR we pass pre-computed net_income grid.
    // Let's assume we pass a pointer to pre-computed net income grid [Nz].
) {
    extern __shared__ double s_mem[];
    double* s_m_endo = s_mem;
    double* s_c_endo = &s_mem[Nm];

    int ia = blockIdx.x % Na;
    int iz = blockIdx.x / Na;
    
    int tid = threadIdx.x;
    if (tid >= Nm) return; // Guard

    double a_curr = d_a_grid[ia];
    // Need Net Income. Let's assume simple y = z for now or passed in.
    // TODO: Pass proper net income array pointer. d_y[iz].
    double net_income = d_z_grid[iz]; // Placeholder! 

    double a_next = a_curr * (1.0 + r_a); // No adjust

    // --- Step 1: EGM (Backward) ---
    // tid corresponds to m_next grid point
    double m_next = d_m_grid[tid];
    
    // Interpolate E_Vm at (m_next, a_next)
    // d_EVm is size (Nm * Na * Nz). Slice offset = iz * (Nm*Na).
    // But we need to interp over 'a' as well.
    double emv = d_interp_2d(&d_EVm[iz * Nm * Na], d_m_grid, Nm, d_a_grid, Na, m_next, a_next);
    
    double rhs = beta * emv;
    double c = d_inv_u_prime(rhs, sigma);
    
    // m_curr = (c + m' - z) / (1+r)
    double m_curr = (c + m_next - net_income) / (1.0 + r_m);

    s_m_endo[tid] = m_curr;
    s_c_endo[tid] = c;

    __syncthreads();

    // --- Step 2: Re-interpolation (Forward) ---
    // tid corresponds to m_fixed (grid)
    double m_fixed = d_m_grid[tid];
    
    // Interp s_c_endo(s_m_endo) at m_fixed
    // Handle constraints
    double c_val, m_prime_val;
    
    if (m_fixed < s_m_endo[0]) {
        // Constrained
        m_prime_val = d_m_grid[0]; // m_min
        c_val = (1.0 + r_m) * m_fixed + net_income - m_prime_val;
    } else {
        // Linear Interp on shared mem
        // We can reuse d_interp_1d logic but need to handle pointer to shared
        // d_interp_1d assumes device pointer? Yes. Shared is device pointer.
        c_val = d_interp_1d(s_m_endo, s_c_endo, Nm, m_fixed);
        m_prime_val = d_interp_1d(s_m_endo, d_m_grid, Nm, m_fixed);
    }
    
    if (c_val < 1e-9) c_val = 1e-9;
    
    // Value Function
    // V = u(c) + beta * E_V(m', a')
    double ev = d_interp_2d(&d_EV[iz * Nm * Na], d_m_grid, Nm, d_a_grid, Na, m_prime_val, a_next);
    double val = d_u(c_val, sigma) + beta * ev;
    
    // Store Result
    int idx = (iz * Na * Nm) + (ia * Nm) + tid;
    d_V[idx] = val;
    d_c[idx] = c_val;
    d_m_pol[idx] = m_prime_val;
    d_a_pol[idx] = a_next;
    d_d_pol[idx] = 0.0;
    d_flag[idx] = 0.0;
}


__global__ void bellman_adjust_kernel(
    double* d_V, double* d_c, double* d_m_pol, double* d_a_pol, double* d_d_pol, double* d_flag,
    const double* d_EV, const double* d_EVm,
    const double* d_m_grid, const double* d_a_grid,
    int Nm, int Na, int Nz,
    double beta, double r_m, double r_a, double sigma, double m_min, double chi,
    const double* d_z_grid
) {
    extern __shared__ double s_mem[];
    double* s_m_endo = s_mem;
    double* s_c_endo = &s_mem[Nm];

    int ia = blockIdx.x % Na;
    int iz = blockIdx.x / Na;
    int tid = threadIdx.x;
    
    if (tid >= Nm) return;
    
    int idx = (iz * Na * Nm) + (ia * Nm) + tid;
    double a_curr = d_a_grid[ia];
    double net_income = d_z_grid[iz];
    double m_fixed = d_m_grid[tid];

    // Initialize best with No-Adjust values (already in global memory)
    double best_v = d_V[idx];
    double best_c = d_c[idx];
    double best_m = d_m_pol[idx];
    double best_a = d_a_pol[idx];
    double best_d = 0.0;
    double best_flag = d_flag[idx]; // should be 0

    // Loop over ALL possible next illiquid assets (a_next)
    for (int ia_next = 0; ia_next < Na; ++ia_next) {
        
        // 1. EGM Construct (Parallel by im_next mapped to tid)
        // Optimization: m_next_grid IS d_m_grid[tid]. 
        // EVm is defined on d_m_grid. So lookup is exact.
        // No need for d_interp_1d.
        
        // Global memory read: Coalesced [iz][Na][Nm] structure?
        // EVm layout: [iz * Na * Nm + ia * Nm + im]
        // Current index: [iz][ia_next][tid]. This IS coalesced (contiguous tids read contiguous memory).
        int evm_idx = (iz * Na * Nm) + (ia_next * Nm) + tid;
        double emv = d_EVm[evm_idx];
        
        double a_next_val = d_a_grid[ia_next]; // Read from global or shared cache
        
        double d = a_next_val - a_curr * (1.0 + r_a);
        double cost = chi * d * d;
        double total_outflow = d + cost;

        double c_e = d_inv_u_prime(beta * emv, sigma);
        // m_next_grid is d_m_grid[tid]
        double m_curr_e = (c_e + d_m_grid[tid] - net_income + total_outflow) / (1.0 + r_m); 
        
        s_m_endo[tid] = m_curr_e;
        s_c_endo[tid] = c_e;
        
        __syncthreads();
        
        // 2. Re-interpolate (Parallel by im_curr mapped to tid)
        // We want value at m_fixed (d_m_grid[tid]).
        // Interpolate (s_m_endo, s_c_endo).
        
        double c_adj, m_prime_adj;
        if (m_fixed < s_m_endo[0]) {
             m_prime_adj = d_m_grid[0];
             c_adj = (1.0 + r_m)*m_fixed + net_income - total_outflow - m_prime_adj;
        } else {
             c_adj = d_interp_1d(s_m_endo, s_c_endo, Nm, m_fixed);
             m_prime_adj = d_interp_1d(s_m_endo, d_m_grid, Nm, m_fixed);
        }
        
        if (c_adj > 1e-9) {
             // Value Calculation
             // Needed: EV(m_prime_adj, a_next).
             // a_next is grid point. So we only interp on m.
             // EV layout: [iz][ia_next][m].
             double ev_val = d_interp_1d(d_m_grid, &d_EV[iz*Na*Nm + ia_next*Nm], Nm, m_prime_adj);
             
             double v_adj = d_u(c_adj, sigma) + beta * ev_val;
             
             if (v_adj > best_v) {
                 best_v = v_adj;
                 best_c = c_adj;
                 best_m = m_prime_adj;
                 best_a = a_next_val;
                 best_d = d;
                 best_flag = 1.0;
             }
        }
        
        __syncthreads();
    }
    
    // final write
    d_V[idx] = best_v;
    d_c[idx] = best_c;
    d_m_pol[idx] = best_m;
    d_a_pol[idx] = best_a;
    d_d_pol[idx] = best_d;
    d_flag[idx] = best_flag;
}

void launch_bellman_kernel(CudaBackend& backend, const std::vector<double>& params) {
    // Unpack Params 
    // [0] beta, [1] rm, [2] ra, [3] sigma, [4] m_min, [5] chi
    double beta = params[0];
    double rm = params[1];
    double ra = params[2];
    double sigma = params[3];
    double m_min = params[4];
    // Check if chi passed. If vector size < 6, default 0 or error.
    // Assuming we update solver to pass chi.
    double chi = 0.0;
    if(params.size() > 5) chi = params[5];
    
    int Nm = backend.N_m;
    int Na = backend.N_a;
    int Nz = backend.N_z;
    
    dim3 blocks(Na * Nz);
    dim3 threads(Nm);
    size_t shared_size = 2 * Nm * sizeof(double);
    
    // 1. No Adjust
    bellman_no_adjust_kernel<<<blocks, threads, shared_size>>>(
        backend.d_V_curr, 
        backend.d_c_pol,
        backend.d_m_pol,
        backend.d_a_pol,
        backend.d_d_pol, 
        backend.d_adjust_flag,
        
        backend.d_EV, backend.d_EVm,
        backend.d_m_grid, backend.d_a_grid,
        Nm, Na, Nz,
        beta, rm, ra, sigma, m_min,
        0.0, backend.d_y_grid, nullptr
    );
    
    // 2. Adjust (Upper Envelope)
    // Run unconditionally to handle both costly and frictionless adjustment cases.
    bellman_adjust_kernel<<<blocks, threads, shared_size>>>(
        backend.d_V_curr, 
        backend.d_c_pol,
        backend.d_m_pol,
        backend.d_a_pol,
        backend.d_d_pol, 
        backend.d_adjust_flag,
        
        backend.d_EV, backend.d_EVm,
        backend.d_m_grid, backend.d_a_grid,
        Nm, Na, Nz,
        beta, rm, ra, sigma, m_min, chi,
        backend.d_y_grid
    );
    
    cudaError_t err = cudaGetLastError();
}

} // namespace Monad

namespace Monad {

// --- Dual Kernel (v3.2) ---

__global__ void bellman_dual_no_adjust_kernel(
    // Outputs (Values + Derivatives)
    double* d_c_val, double* d_c_der,
    double* d_m_val, double* d_m_der,
    double* d_a_val, double* d_a_der,
    
    // Inputs (Standard)
    const double* d_EV, const double* d_EVm,
    const double* d_m_grid, const double* d_a_grid,
    int Nm, int Na, int Nz,
    double beta, 
    Duald r_m, Duald r_a, // Dual Seeds
    double sigma, double m_min,
    const double* d_z_grid,
    int shock_mode
) {
    extern __shared__ Duald s_dual_mem[]; // Typed as Duald
    Duald* s_m_endo = s_dual_mem;
    Duald* s_c_endo = &s_dual_mem[Nm];

    int ia = blockIdx.x % Na;
    int iz = blockIdx.x / Na;
    int tid = threadIdx.x;
    
    if (tid >= Nm) return;

    // Dual Setup
    Duald a_curr(d_a_grid[ia], 0.0);
    double net_income_val = d_z_grid[iz];
    // Income Shock Logic: If shock_mode=1, derivative is proportional to income (z_i)
    // This assumes unit shock dZ/Z = 1 (or dlogZ = 1) scales all income by 1%
    double income_der = (shock_mode == 1) ? net_income_val : 0.0;
    Duald net_income(net_income_val, income_der);

    Duald a_next = a_curr * (1.0 + r_a); // Dual propagation
    
    // --- Step 1: EGM (Backward) ---
    // m_next is grid point (double), but we treat as Dual(val, 0)
    double m_next_val = d_m_grid[tid];
    Duald m_next(m_next_val, 0.0);
    
    // Interpolate E_Vm at (m_next, a_next). a_next is Dual.
    // Use templated interp.
    // EVm is double array.
    Duald emv = d_interp_2d(d_EVm + iz * Nm * Na, d_m_grid, Nm, d_a_grid, Na, m_next, a_next);
    
    Duald rhs = beta * emv;
    Duald c = d_inv_u_prime(rhs, sigma);
    
    // m_curr = (c + m' - z) / (1+r_m)
    Duald m_curr = (c + m_next - net_income) / (1.0 + r_m);

    s_m_endo[tid] = m_curr;
    s_c_endo[tid] = c;

    __syncthreads();

    // --- Step 2: Re-interpolation (Forward) ---
    double m_fixed_val = d_m_grid[tid];
    Duald m_fixed(m_fixed_val, 0.0);
    
    Duald c_val, m_prime_val;
    
    if (m_fixed < s_m_endo[0]) {
        // Constrained
        m_prime_val = Duald(d_m_grid[0], 0.0);
        c_val = (1.0 + r_m) * m_fixed + net_income - m_prime_val;
    } else {
        // Custom interpolation on shared memory (Duald arrays)
        int left = 0; 
        int right = Nm - 1;
        while(left < right - 1) {
             int mid = (left + right)/2;
             if (s_m_endo[mid] <= m_fixed) left = mid; else right = mid;
        }
        
        Duald t = (m_fixed - s_m_endo[left]) / (s_m_endo[right] - s_m_endo[left]);
        c_val = s_c_endo[left] + t * (s_c_endo[right] - s_c_endo[left]);
        
        // m_prime is the m_next that generated this m_curr (inverse EGM mapping)
        // d_m_grid points are doubles
        double y_left = d_m_grid[left];
        double y_right = d_m_grid[right];
        
        // Linear mix using Dual 't'
        // y_left + t * (y_right - y_left)
        m_prime_val = t * (y_right - y_left) + y_left;
    }
    
    if (c_val.val < 1e-9) c_val = Duald(1e-9, 0.0);
    
    // Store Results
    int idx = (iz * Na * Nm) + (ia * Nm) + tid;
    
    d_c_val[idx] = c_val.val;
    d_c_der[idx] = c_val.der;
    
    d_m_val[idx] = m_prime_val.val;
    d_m_der[idx] = m_prime_val.der;
    
    d_a_val[idx] = a_next.val;
    d_a_der[idx] = a_next.der;
}

void launch_bellman_dual(CudaBackend& backend, double r_m_val, double r_a_val, 
                        double seed_rm, double seed_ra,
                        double* d_out_c_der, double* d_out_m_der, double* d_out_a_der,
                        int shock_mode) {
    
    // Determine seeds based on Shock Mode
    // Mode 0: Interest Rate Shock (r_m moves)
    // Mode 1: Income Shock (r_m fixed, Z moves)
    double s_rm = (shock_mode == 0) ? 1.0 : 0.0;
    double s_ra = 0.0; // r_a assumed fixed for now
    
    Duald r_m(r_m_val, s_rm);
    Duald r_a(r_a_val, s_ra);
    
    int Nm = backend.N_m;
    int Na = backend.N_a;
    int Nz = backend.N_z;
    
    dim3 blocks(Na * Nz);
    dim3 threads(Nm);
    
    // Shared memory size for Duald arrays (2 * Nm * sizeof(Duald))
    // Duald is 2 doubles (16 bytes)
    size_t shared_size = 2 * Nm * sizeof(Duald);

    // We need standard inputs (EV, EVm, grids)
    double sigma = 1.0; // Fixed for now or pass in
    double m_min = 0.0;
    double beta = 0.986; // TO FIX: Pass full params vector
    
    bellman_dual_no_adjust_kernel<<<blocks, threads, shared_size>>>(
        backend.d_c_pol, d_out_c_der,
        backend.d_m_pol, d_out_m_der,
        backend.d_a_pol, d_out_a_der,
        
        backend.d_EV, backend.d_EVm,
        backend.d_m_grid, backend.d_a_grid,
        Nm, Na, Nz,
        beta, r_m, r_a, sigma, m_min,
        backend.d_y_grid, // Assuming z grid passed here for net income
        shock_mode 
    );
    
    cudaError_t err = cudaGetLastError();
}

// --- FakeNews Kernel (v3.2 Step 3) ---
// Computes F_r = sum_i D[i] * dW_ij / dr
// where dW_ij/dr = dW/dm' * dm'/dr + dW/da' * da'/dr

__global__ void fake_news_kernel(
    double* d_F,               // Output: Fake news vector [N_total]
    const double* d_D,         // Input: Distribution [N_total]
    const double* d_m_pol,     // Input: m' policy [N_total]
    const double* d_a_pol,     // Input: a' policy [N_total]
    const double* d_m_der,     // Input: dm'/dr [N_total]
    const double* d_a_der,     // Input: da'/dr [N_total]
    const double* d_m_grid,
    const double* d_a_grid,
    int Nm, int Na, int Nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nz;
    
    if (idx >= total) return;
    
    double mass = d_D[idx];
    if (mass < 1e-12) return; // Skip negligible mass
    
    // Get policy values and derivatives
    double m_prime = d_m_pol[idx];
    double a_prime = d_a_pol[idx];
    double dm_dr = d_m_der[idx];
    double da_dr = d_a_der[idx];
    
    // Decompose idx to get iz
    int iz = idx / (Nm * Na);
    
    // Find m interpolation indices
    int im_lo = 0;
    for (int i = 0; i < Nm - 1; ++i) {
        if (d_m_grid[i + 1] > m_prime) break;
        im_lo = i;
    }
    int im_hi = min(im_lo + 1, Nm - 1);
    
    double dm_grid = d_m_grid[im_hi] - d_m_grid[im_lo];
    double t_m = (dm_grid > 1e-10) ? (m_prime - d_m_grid[im_lo]) / dm_grid : 0.0;
    t_m = max(0.0, min(1.0, t_m));
    
    // dW_m / dm' = d(1-t_m)/dm' = -1/dm_grid for low, +1/dm_grid for high
    double dW_m_lo = -1.0 / dm_grid; // dW_lo/dm'
    double dW_m_hi = +1.0 / dm_grid; // dW_hi/dm'
    
    // Find a interpolation indices  
    int ia_lo = 0;
    for (int i = 0; i < Na - 1; ++i) {
        if (d_a_grid[i + 1] > a_prime) break;
        ia_lo = i;
    }
    int ia_hi = min(ia_lo + 1, Na - 1);
    
    double da_grid = d_a_grid[ia_hi] - d_a_grid[ia_lo];
    double t_a = (da_grid > 1e-10) ? (a_prime - d_a_grid[ia_lo]) / da_grid : 0.0;
    t_a = max(0.0, min(1.0, t_a));
    
    double dW_a_lo = -1.0 / da_grid;
    double dW_a_hi = +1.0 / da_grid;
    
    // Bilinear weights: W_ij = (1-t_m)(1-t_a), t_m(1-t_a), (1-t_m)t_a, t_m*t_a
    // dW/dr = dW/dm' * dm'/dr + dW/da' * da'/dr
    
    // Four corners: (im_lo, ia_lo), (im_hi, ia_lo), (im_lo, ia_hi), (im_hi, ia_hi)
    double w00 = (1.0 - t_m) * (1.0 - t_a);
    double w10 = t_m * (1.0 - t_a);
    double w01 = (1.0 - t_m) * t_a;
    double w11 = t_m * t_a;
    
    // dW/dr for each corner
    // d[(1-t_m)(1-t_a)]/dr = -(1-t_a)*dt_m/dr - (1-t_m)*dt_a/dr
    // dt_m/dr = (1/dm_grid) * dm'/dr
    // dt_a/dr = (1/da_grid) * da'/dr
    
    double dt_m_dr = dm_dr / dm_grid;
    double dt_a_dr = da_dr / da_grid;
    
    double dw00 = -(1.0 - t_a) * dt_m_dr - (1.0 - t_m) * dt_a_dr;
    double dw10 = (1.0 - t_a) * dt_m_dr - t_m * dt_a_dr;
    double dw01 = -t_a * dt_m_dr + (1.0 - t_m) * dt_a_dr;
    double dw11 = t_a * dt_m_dr + t_m * dt_a_dr;
    
    // Scatter with atomicAdd
    int idx00 = iz * (Na * Nm) + ia_lo * Nm + im_lo;
    int idx10 = iz * (Na * Nm) + ia_lo * Nm + im_hi;
    int idx01 = iz * (Na * Nm) + ia_hi * Nm + im_lo;
    int idx11 = iz * (Na * Nm) + ia_hi * Nm + im_hi;
    
    // Use atomicAdd for double (requires compute capability 6.0+)
    atomicAdd(&d_F[idx00], mass * dw00);
    atomicAdd(&d_F[idx10], mass * dw10);
    atomicAdd(&d_F[idx01], mass * dw01);
    atomicAdd(&d_F[idx11], mass * dw11);
}

void launch_fake_news(CudaBackend& backend, 
                     const double* d_D,
                     double* d_F) {
    int total = backend.N_m * backend.N_a * backend.N_z;
    
    // Zero out F
    cudaMemset(d_F, 0, total * sizeof(double));
    
    dim3 blocks((total + 255) / 256);
    dim3 threads(256);
    
    fake_news_kernel<<<blocks, threads>>>(
        d_F, d_D,
        backend.d_m_pol, backend.d_a_pol,
        backend.d_m_der, backend.d_a_der,
        backend.d_m_grid, backend.d_a_grid,
        backend.N_m, backend.N_a, backend.N_z
    );
    
    cudaError_t err = cudaGetLastError();
}

// --- GPU Weighted Sum Reduction (v3.2 Step 4) ---
// Computes: result = sum_i (a[i] * b[i])
// Uses parallel reduction with shared memory

__global__ void weighted_sum_kernel(
    const double* a,
    const double* b,
    double* partial_sums,
    int n
) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and multiply
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

double gpu_weighted_sum(const double* d_a, const double* d_b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Allocate partial sums
    double* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(double));
    
    // First reduction
    weighted_sum_kernel<<<blocks, threads, threads * sizeof(double)>>>(
        d_a, d_b, d_partial, n
    );
    
    // Download partial sums and finish on CPU
    std::vector<double> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);
    
    double total = 0.0;
    for (double v : h_partial) total += v;
    
    return total;
}

// Forward iteration of distribution perturbation on GPU
// dD_{t+1} = Lambda^T * dD_t
// For simplicity, we use F directly (first-order effect) and iterate
__global__ void dist_forward_kernel(
    double* d_dD_next,
    const double* d_dD,
    const double* d_m_pol,
    const double* d_a_pol,
    const double* d_m_grid,
    const double* d_a_grid,
    int Nm, int Na, int Nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nm * Na * Nz;
    
    if (idx >= total) return;
    
    double mass = d_dD[idx];
    if (fabs(mass) < 1e-15) return;
    
    double m_prime = d_m_pol[idx];
    double a_prime = d_a_pol[idx];
    int iz = idx / (Nm * Na);
    
    // Find interpolation indices (same as fake_news_kernel)
    int im_lo = 0;
    for (int i = 0; i < Nm - 1; ++i) {
        if (d_m_grid[i + 1] > m_prime) break;
        im_lo = i;
    }
    int im_hi = min(im_lo + 1, Nm - 1);
    
    double dm_grid = d_m_grid[im_hi] - d_m_grid[im_lo];
    double t_m = (dm_grid > 1e-10) ? (m_prime - d_m_grid[im_lo]) / dm_grid : 0.0;
    t_m = max(0.0, min(1.0, t_m));
    
    int ia_lo = 0;
    for (int i = 0; i < Na - 1; ++i) {
        if (d_a_grid[i + 1] > a_prime) break;
        ia_lo = i;
    }
    int ia_hi = min(ia_lo + 1, Na - 1);
    
    double da_grid = d_a_grid[ia_hi] - d_a_grid[ia_lo];
    double t_a = (da_grid > 1e-10) ? (a_prime - d_a_grid[ia_lo]) / da_grid : 0.0;
    t_a = max(0.0, min(1.0, t_a));
    
    // Bilinear weights
    double w00 = (1.0 - t_m) * (1.0 - t_a);
    double w10 = t_m * (1.0 - t_a);
    double w01 = (1.0 - t_m) * t_a;
    double w11 = t_m * t_a;
    
    // Scatter
    int idx00 = iz * (Na * Nm) + ia_lo * Nm + im_lo;
    int idx10 = iz * (Na * Nm) + ia_lo * Nm + im_hi;
    int idx01 = iz * (Na * Nm) + ia_hi * Nm + im_lo;
    int idx11 = iz * (Na * Nm) + ia_hi * Nm + im_hi;
    
    atomicAdd(&d_dD_next[idx00], mass * w00);
    atomicAdd(&d_dD_next[idx10], mass * w10);
    atomicAdd(&d_dD_next[idx01], mass * w01);
    atomicAdd(&d_dD_next[idx11], mass * w11);
}

void launch_dist_forward(CudaBackend& backend, double* d_dD, double* d_dD_next) {
    int total = backend.N_m * backend.N_a * backend.N_z;
    
    cudaMemset(d_dD_next, 0, total * sizeof(double));
    
    dim3 blocks((total + 255) / 256);
    dim3 threads(256);
    
    dist_forward_kernel<<<blocks, threads>>>(
        d_dD_next, d_dD,
        backend.d_m_pol, backend.d_a_pol,
        backend.d_m_grid, backend.d_a_grid,
        backend.N_m, backend.N_a, backend.N_z
    );
}

// Full IRF Computation on GPU - v3.3 Extended with dB
IRFResult compute_irf_gpu(CudaBackend& backend, int T, int shock_mode) {
    int total = backend.N_m * backend.N_a * backend.N_z;
    
    IRFResult result;
    result.dC.resize(T);
    result.dB.resize(T);
    
    // Allocate temp buffer for forward iteration
    double* d_dD_temp;
    cudaMalloc(&d_dD_temp, total * sizeof(double));
    
    // 4. Compute Jacobian of Policy (Backward)
    // Need current params r_m, r_a.
    // For now assuming r_m = 0.01 (1%), r_a = 0.0.
    // TO FIX: Should pull from Params or Backend state.
    // This step does NOT depend on t, only on SS. So do it ONCE provided backend has SS policy.
    static bool deriv_computed = false;
    // Actually for different shock modes we need to recompute.
    // Let's recompute always to be safe.
    
    // Passed shock_mode determines the seed
    launch_bellman_dual(backend, 0.01, 0.0, 1.0, 0.0, 
                        backend.d_c_der, backend.d_m_der, backend.d_a_der, shock_mode);
        
    // 5. Compute Fake News Vector (F_0) (Forward Impulses)
    launch_fake_news(backend, backend.d_D, backend.d_F);
    
    // Start with F as dD_0 (impact effect)
    // d_F already contains F from fake_news_kernel
    
    for (int t = 0; t < T; ++t) {
        // Compute dC_t = sum_i dD_t[i] * c[i]
        double dC_t = gpu_weighted_sum(backend.d_F, backend.d_c_pol, total);
        result.dC[t] = dC_t;
        
        // v3.3: Compute dB_t = sum_i dD_t[i] * m_pol[i] (liquid asset response)
        double dB_t = gpu_weighted_sum(backend.d_F, backend.d_m_pol, total);
        result.dB[t] = dB_t;
        
        // Forward iterate: dD_{t+1} = Lambda^T * dD_t
        if (t < T - 1) {
            launch_dist_forward(backend, backend.d_F, d_dD_temp);
            cudaMemcpy(backend.d_F, d_dD_temp, total * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }
    
    cudaFree(d_dD_temp);
    
    return result;
}

} // namespace Monad (Re-opened)
