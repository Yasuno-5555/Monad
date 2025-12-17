#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>

// Macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

namespace Monad {

class CudaBackend {
public:
    // Device Pointers (VRAM)
    double* d_m_grid = nullptr;
    double* d_a_grid = nullptr;
    double* d_y_grid = nullptr; // Net Income Grid
    double* d_V_curr = nullptr;
    double* d_V_next = nullptr;
    double* d_Pi     = nullptr; 
    
    // Policy Functions
    double* d_c_pol  = nullptr;
    double* d_m_pol  = nullptr;
    double* d_a_pol  = nullptr;
    double* d_d_pol  = nullptr;
    double* d_adjust_flag = nullptr;
    
    // Expectation Buffers
    double* d_EV     = nullptr;
    double* d_EVm    = nullptr;
    
    // Jacobian Pointers
    double* d_c_der  = nullptr;
    double* d_m_der  = nullptr;
    double* d_a_der  = nullptr;
    
    // FakeNews / SSJ Buffers
    double* d_D      = nullptr; // Distribution
    double* d_F      = nullptr; // FakeNews Vector
    
    // Grid Dimensions
    int N_m, N_a, N_z;

    CudaBackend(int nm, int na, int nz) : N_m(nm), N_a(na), N_z(nz) {
        initialize_memory();
    }

    ~CudaBackend() {
        free_memory();
    }

    // Allocate memory on GPU
    void initialize_memory() {
        size_t grid_size_m = N_m * sizeof(double);
        size_t grid_size_a = N_a * sizeof(double);
        size_t total_size = N_m * N_a * N_z * sizeof(double);

        CUDA_CHECK(cudaMalloc((void**)&d_m_grid, grid_size_m));
        CUDA_CHECK(cudaMalloc((void**)&d_a_grid, grid_size_a));
        CUDA_CHECK(cudaMalloc((void**)&d_y_grid, N_z * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_V_curr, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_V_next, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_c_pol,  total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_m_pol,  total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_a_pol,  total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_d_pol,  total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_adjust_flag, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_Pi, N_z * N_z * sizeof(double)));
        
        // Output Buffers
        CUDA_CHECK(cudaMalloc((void**)&d_EV, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_EVm, total_size));
        
        // Jacobian Buffers (v3.2)
        CUDA_CHECK(cudaMalloc((void**)&d_c_der, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_m_der, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_a_der, total_size));
        
        // FakeNews / SSJ Buffers (v3.2)
        CUDA_CHECK(cudaMalloc((void**)&d_D, total_size));
        CUDA_CHECK(cudaMalloc((void**)&d_F, total_size));
        
        std::cout << "[CUDA] Allocated " << (total_size * 2 + grid_size_m + grid_size_a) / 1024 / 1024 
                  << " MB on GPU." << std::endl;
    }

    // Copy Data: Host -> Device
    void upload_grids(const std::vector<double>& h_m, const std::vector<double>& h_a) {
        CUDA_CHECK(cudaMemcpy(d_m_grid, h_m.data(), N_m * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a_grid, h_a.data(), N_a * sizeof(double), cudaMemcpyHostToDevice));
    }

    void upload_income(const std::vector<double>& h_y) {
         CUDA_CHECK(cudaMemcpy(d_y_grid, h_y.data(), N_z * sizeof(double), cudaMemcpyHostToDevice));
    }

    void upload_value(const std::vector<double>& h_V) {
        CUDA_CHECK(cudaMemcpy(d_V_next, h_V.data(), N_m * N_a * N_z * sizeof(double), cudaMemcpyHostToDevice));
    }

    void upload_pi(const std::vector<double>& h_Pi) {
        CUDA_CHECK(cudaMemcpy(d_Pi, h_Pi.data(), N_z * N_z * sizeof(double), cudaMemcpyHostToDevice));
    }

    void upload_policy(const std::vector<double>& h_c) {
        CUDA_CHECK(cudaMemcpy(d_c_pol, h_c.data(), N_m * N_a * N_z * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    // Download Expectation Results
    void download_expectations(std::vector<double>& h_EV, std::vector<double>& h_EVm) {
        CUDA_CHECK(cudaMemcpy(h_EV.data(), d_EV, N_m * N_a * N_z * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_EVm.data(), d_EVm, N_m * N_a * N_z * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Download Full Policy
    void download_policy(std::vector<double>& h_c, std::vector<double>& h_m, 
                         std::vector<double>& h_a, std::vector<double>& h_d, 
                         std::vector<double>& h_flag) {
        size_t size = N_m * N_a * N_z * sizeof(double);
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c_pol, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_m.data(), d_m_pol, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_a.data(), d_a_pol, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_d.data(), d_d_pol, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flag.data(), d_adjust_flag, size, cudaMemcpyDeviceToHost));
    }

    // Copy Data: Device -> Host (Retrieve results)
    void download_value(std::vector<double>& h_V) {
        CUDA_CHECK(cudaMemcpy(h_V.data(), d_V_curr, N_m * N_a * N_z * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Basic Warmup / Test
    void verify_device() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "[CUDA] Running on: " << prop.name << std::endl;
        std::cout << "[CUDA] VRAM: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
    }

private:
    void free_memory() {
        if (d_m_grid) cudaFree(d_m_grid);
        if (d_a_grid) cudaFree(d_a_grid);
        if (d_y_grid) cudaFree(d_y_grid);
        if (d_V_curr) cudaFree(d_V_curr);
        if (d_V_next) cudaFree(d_V_next);
        if (d_c_pol)  cudaFree(d_c_pol);
        if (d_m_pol)  cudaFree(d_m_pol);
        if (d_a_pol)  cudaFree(d_a_pol);
        if (d_d_pol)  cudaFree(d_d_pol);
        if (d_adjust_flag) cudaFree(d_adjust_flag);
        if (d_Pi) cudaFree(d_Pi);
        if (d_EV) cudaFree(d_EV);
        if (d_EVm) cudaFree(d_EVm);
        if (d_c_der) cudaFree(d_c_der);
        if (d_m_der) cudaFree(d_m_der);
        if (d_a_der) cudaFree(d_a_der);
        if (d_D) cudaFree(d_D);
        if (d_F) cudaFree(d_F);
    }
};

// Kernel Wrappers
void launch_bellman_kernel(CudaBackend& backend, const std::vector<double>& params);
void launch_expectations(CudaBackend& backend, double r_m, double sigma);
void launch_bellman_dual(CudaBackend& backend, double r_m_val, double r_a_val, 
                        double seed_rm, double seed_ra,
                        double* d_out_c_der, double* d_out_m_der, double* d_out_a_der);
void launch_fake_news(CudaBackend& backend, const double* d_D, double* d_F);
std::vector<double> compute_irf_gpu(CudaBackend& backend, int T);

} // namespace Monad
