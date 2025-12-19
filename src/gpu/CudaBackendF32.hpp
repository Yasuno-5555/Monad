#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>

// Phase 5: Float32 Backend for 8GB VRAM Optimization
// Replaces 'double' with 'float' (Real)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

namespace Monad {

// Optimization: Single Precision
using Real = float;

class CudaBackendF32 {
public:
    // Device Pointers (VRAM) - Float32
    Real* d_m_grid = nullptr;
    Real* d_as_grid = nullptr; // Stock
    Real* d_ac_grid = nullptr; // Crypto
    Real* d_y_grid = nullptr;
    
    // 5D Arrays
    Real* d_V_curr = nullptr;
    Real* d_V_next = nullptr;
    
    // Policy (m', as', ac')
    Real* d_m_pol  = nullptr;
    Real* d_as_pol = nullptr;
    Real* d_ac_pol = nullptr;
    Real* d_c_pol  = nullptr; // Consumption
    
    // Distribution & Temp
    Real* d_D      = nullptr; // Distribution
    Real* d_D_next = nullptr;
    
    // Grid Dimensions
    int N_m, N_as, N_ac, N_z, N_b;

    CudaBackendF32(int nm, int nas, int nac, int nz, int nb) 
        : N_m(nm), N_as(nas), N_ac(nac), N_z(nz), N_b(nb) {
        initialize_memory();
    }

    ~CudaBackendF32() {
        free_memory();
    }

    void initialize_memory() {
        size_t total_nodes = (size_t)N_m * N_as * N_ac * N_z * N_b;
        size_t memory_bytes = total_nodes * sizeof(Real);

        // Assets Layout: (b, z, ac, as, m)
        CUDA_CHECK(cudaMalloc((void**)&d_m_grid, N_m * sizeof(Real)));
        CUDA_CHECK(cudaMalloc((void**)&d_as_grid, N_as * sizeof(Real)));
        CUDA_CHECK(cudaMalloc((void**)&d_ac_grid, N_ac * sizeof(Real)));
        CUDA_CHECK(cudaMalloc((void**)&d_y_grid, N_z * sizeof(Real)));
        
        // 5D State Arrays
        CUDA_CHECK(cudaMalloc((void**)&d_V_curr, memory_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_V_next, memory_bytes));
        
        CUDA_CHECK(cudaMalloc((void**)&d_m_pol,  memory_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_as_pol, memory_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_ac_pol, memory_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_c_pol,  memory_bytes));
        
        CUDA_CHECK(cudaMalloc((void**)&d_D,      memory_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_D_next, memory_bytes));
        
        float vram_mb = (float)(memory_bytes * 8 + N_m*4 + N_as*4 + N_ac*4) / 1024.0f / 1024.0f;
        std::cout << "[CUDA F32] Allocated " << vram_mb << " MB for 5D Grid (" << total_nodes << " states)." << std::endl;
        
        // Example: 20*20*20*5*5 = 200,000 states.
        // 200,000 * 4 bytes = 0.8 MB per array.
        // Total ~8 arrays * 0.8 = 6.4 MB.
        // TINY! Even 8GB card can handle 1000x this.
        // We could easily increase grid density.
    }
    
    // Data Transfer Helpers
    void upload_grids(const std::vector<float>& h_m, const std::vector<float>& h_as, const std::vector<float>& h_ac) {
        CUDA_CHECK(cudaMemcpy(d_m_grid, h_m.data(), N_m * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_as_grid, h_as.data(), N_as * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ac_grid, h_ac.data(), N_ac * sizeof(Real), cudaMemcpyHostToDevice));
    }

private:
    void free_memory() {
        if (d_m_grid) cudaFree(d_m_grid);
        if (d_as_grid) cudaFree(d_as_grid);
        if (d_ac_grid) cudaFree(d_ac_grid);
        if (d_y_grid) cudaFree(d_y_grid);
        
        if (d_V_curr) cudaFree(d_V_curr);
        if (d_V_next) cudaFree(d_V_next);
        
        if (d_m_pol) cudaFree(d_m_pol);
        if (d_as_pol) cudaFree(d_as_pol);
        if (d_ac_pol) cudaFree(d_ac_pol);
        if (d_c_pol) cudaFree(d_c_pol);
        
        if (d_D) cudaFree(d_D);
        if (d_D_next) cudaFree(d_D_next);
    }
};

} // namespace Monad
