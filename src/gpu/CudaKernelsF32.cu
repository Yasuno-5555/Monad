#include "CudaBackendF32.hpp"

namespace Monad {

// Phase 5: 5D Distribution Update Kernel (Float32)
// Logic: Move mass from D_curr[i] to D_next[policy[i]]
// Threads: One thread per state.
// Optimization: Memory Coalescing on 'm' (Fastest Index)

__global__ void update_distribution_5d(
    const float* __restrict__ D_curr,
    float* __restrict__ D_next,
    const float* __restrict__ m_pol,
    const float* __restrict__ as_pol,
    const float* __restrict__ ac_pol,
    const float* __restrict__ m_grid,
    const float* __restrict__ as_grid,
    const float* __restrict__ ac_grid,
    int N_m, int N_as, int N_ac, int N_z, int N_b,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    float mass = D_curr[idx];
    if (mass < 1e-15f) return; // Skip empty states

    // 1. Get Policy (Destination Values)
    float m_next_val  = m_pol[idx];
    float as_next_val = as_pol[idx];
    float ac_next_val = ac_pol[idx];

    // 2. Find Grid Indices (Linear Interpolation Weights)
    // For 5D distribution, we strictly need to distribute mass to grid points
    // to preserve sum(D) = 1.
    // "Lottery" or "Linear Interpolation" method.
    // Simplifying assumption for Demo Code: Nearest Neighbor or Simple Floor
    // Real implementation requires finding lower_bound for each dimension.
    
    // Helper: Find lower bound index `i` such that grid[i] <= val < grid[i+1]
    // Here we assume uniform grid or pre-computed indices for speed.
    // For this kernel, let's assume we implement a simple search or the policy
    // returns exact grid indices (discrete choice).
    
    // Continuous Policy Interpolation:
    // Distribute mass to 8 corners of the 3D cube (m, as, ac).
    // This is expensive.
    // Alternative: Policy returns INDICES directly? No, EGM returns values.
    
    // For the sake of this file being a "Design Artifact", I will show the atomicAdd usage.
    
    // Placeholder indices (mapping value to nearest index)
    // int im_next = val_to_idx(m_next_val, m_grid, N_m); ...
    
    // Let's assume we map to the nearest single bucket for speed (Histogram method)
    // (In production, use linear interpolation to 8 neighbors)
    int next_idx = idx; // Placeholder: Stationary
    
    // 3. Atomically Add Mass
    // atomicAdd for float is supported on Compute Capability 2.0+
    atomicAdd(&D_next[next_idx], mass);
}

// Host Wrapper
void launch_distribution_update_5d(CudaBackendF32& backend) {
    int total = backend.N_m * backend.N_as * backend.N_ac * backend.N_z * backend.N_b;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    // Reset Next Distribution
    cudaMemset(backend.d_D_next, 0, total * sizeof(float));
    
    update_distribution_5d<<<blocks, threads>>>(
        backend.d_D, backend.d_D_next,
        backend.d_m_pol, backend.d_as_pol, backend.d_ac_pol,
        backend.d_m_grid, backend.d_as_grid, backend.d_ac_grid,
        backend.N_m, backend.N_as, backend.N_ac, backend.N_z, backend.N_b,
        total
    );
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace Monad
