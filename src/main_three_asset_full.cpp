#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>

#include "blocks/IncomeProcessFactory.hpp"
#include "Params.hpp"
#include "grid/MultiDimGrid.hpp"
#include "kernel/TwoAssetKernel.hpp"
#include "backend/gpu/GpuBackend.cuh"

int main(int argc, char** argv) {
    std::cout << "=== Phase G8-A: Scaled Three-Asset Model (GPU Only) ===" << std::endl;
    
    // Grid Setup: Large Scale
    // Target: ~2.5M states -> Fits easily in VRAM (2.5M * 8 bytes * ~10 arrays = ~200MB)
    int nm = 80;
    int na = 80;
    int nh = 40;
    int nz = 5;
    
    std::cout << "Grid Configuration:" << std::endl;
    std::cout << "  N_m: " << nm << std::endl;
    std::cout << "  N_a: " << na << std::endl;
    std::cout << "  N_h: " << nh << std::endl;
    std::cout << "  N_z: " << nz << std::endl;
    long long total_size = (long long)nm * na * nh * nz;
    std::cout << "  Total States: " << total_size << std::endl;

    // --- 1. Host Initialization ---
    std::vector<double> m_nodes(nm), a_nodes(na), h_nodes(nh);
    
    // Non-linear grid for assets to capture curvature near 0
    double m_max = 200.0;
    for(int i = 0; i < nm; ++i) m_nodes[i] = m_max * std::pow((double)i/(nm-1), 2.0); // Quadratic
    
    double a_max = 500.0;
    for(int i = 0; i < na; ++i) a_nodes[i] = a_max * std::pow((double)i/(na-1), 1.5); 
    
    double h_max = 500.0;
    for(int i = 0; i < nh; ++i) h_nodes[i] = h_max * std::pow((double)i/(nh-1), 1.5);

    auto process = Monad::IncomeProcessFactory::make_rouwenhorst(0.966, 0.5, nz);

    Monad::ThreeAssetParam params;
    params.beta = 0.986;
    params.r_m = -0.01;
    params.r_a = 0.02;
    params.r_h = 0.035; // Slightly higher return for H
    params.sigma = 2.0;
    params.chi0 = 0.05; // Fixed Cost
    params.chi1 = 0.05; // Quadratic Cost
    params.chi2 = 0.0;
    
    // Complexity: Wealth Tax
    params.fiscal.tax_rule.cgt_rate = 0.20; 
    params.wealth_tax_rate = 0.01;   // 1% Wealth Tax
    params.wealth_tax_thresh = 100.0; // Threshold

    // Host buffers for result retrieval (minimal)
    std::vector<double> h_V(total_size);
    std::vector<double> h_c(total_size, 0.5); // Initial guess c=0.5
    
    // Initial Guess for V = u(c)/(1-beta)
    for(int i=0; i<total_size; ++i) {
        h_V[i] = (std::pow(h_c[i], 1.0-params.sigma)/(1.0-params.sigma)) / (1.0-params.beta);
    }

    // --- 2. GPU Setup ---
#ifdef MONAD_GPU
    Monad::GpuBackend gpu(nm, na, nh, nz);
    
    std::cout << "Uploading grids to GPU..." << std::endl;
    gpu.upload_grids(m_nodes.data(), a_nodes.data(), h_nodes.data(), process.z_grid.data(), 
                     process.Pi_flat.data(), nm, na, nh, nz);
                     
    gpu.set_params(params.beta, params.r_m, params.r_a, params.r_h, 
                   params.chi0, params.chi1, params.chi2, params.sigma,
                   params.fiscal.tax_rule.lambda, params.fiscal.tax_rule.tau, 
                   params.fiscal.tax_rule.transfer, params.fiscal.tax_rule.cgt_rate,
                   params.wealth_tax_rate, params.wealth_tax_thresh);
    
    // Placeholders for output (we strictly don't need full policy on host every iter, but API requires pointers)
    // To save PCIe bandwidth, we could modify backend to not copy back, but for now let's just stick to API.
    std::vector<double> h_V_new(total_size);
     std::vector<double> h_c_new(total_size);
    std::vector<double> h_m_new(total_size);
    std::vector<double> h_a_new(total_size);
    std::vector<double> h_h_new(total_size);
    std::vector<double> dummy(total_size); // For others

    std::cout << "Starting Bellman Iterations..." << std::endl;
    std::cout << "Iter | Error (|V - V'|) | Time (ms)" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    double error = 1.0;
    int iter = 0;
    int max_iter = 2000;
    double tol = 1e-7; // Convergence tolerance

    auto total_start = std::chrono::high_resolution_clock::now();

    while(error > tol && iter < max_iter) {
        auto start = std::chrono::high_resolution_clock::now();
        
        gpu.solve_bellman_iteration(
            h_V.data(), h_c.data(),
            h_V_new.data(), h_c_new.data(),
            h_m_new.data(), h_a_new.data(), h_h_new.data(),
            dummy.data(), dummy.data(), dummy.data(),
            total_size
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Compute Error (Max Norm) on Host
        // Note: For extreme performance, this reduction should be on GPU.
        // But copying 20MB is fast (~1-2ms).
        error = 0.0;
        for(size_t i=0; i<total_size; ++i) {
            double diff = std::abs(h_V[i] - h_V_new[i]);
            if(diff > error) error = diff;
        }

        if(iter % 10 == 0) {
            std::cout << std::setw(4) << iter << " | " 
                      << std::scientific << std::setprecision(4) << error << " | " 
                      << std::fixed << std::setw(5) << ms << std::endl;
        }

        // Update
        std::swap(h_V, h_V_new);
        std::copy(h_c_new.begin(), h_c_new.end(), h_c.begin()); // Update policy guess if needed (often V is enough)

        iter++;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "-----------------------------------" << std::endl;
    if(error <= tol) {
        std::cout << "CONVERGED in " << iter << " iterations." << std::endl;
        
        // --- Phase G9: Monte Carlo Simulation ---
        std::cout << "\n=== Starting Monte Carlo Simulation ===" << std::endl;
        int n_sim = 10000;
        int t_sim = 1000;
        std::vector<double> hist;
        
        auto sim_start = std::chrono::high_resolution_clock::now();
        gpu.simulate(n_sim, t_sim, hist);
        auto sim_end = std::chrono::high_resolution_clock::now();
        long long sim_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sim_end - sim_start).count();
        
        std::cout << "Simulation Time: " << sim_ms << " ms" << std::endl;
        std::cout << "Mean Wealth (End): " << hist.back() << std::endl;

    } else {
        std::cout << "FAILED to converge within " << max_iter << " iterations." << std::endl;
    }
    std::cout << "Total Runtime: " << total_ms / 1000.0 << " s" << std::endl;
    std::cout << "Average Time/Iter: " << (double)total_ms / iter << " ms" << std::endl;
    
#else
    std::cout << "Error: MONAD_GPU not defined. This verification is GPU only." << std::endl;
#endif

    return 0;
}
