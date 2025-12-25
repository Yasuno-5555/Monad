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
#include "solver/TwoAssetSolver.hpp"
#include "backend/cpu/CpuBackend.hpp"

#ifdef MONAD_GPU
#include "backend/gpu/GpuBackend.cuh"
#endif

int main(int argc, char** argv) {
    std::cout << "=== G7: Three-Asset Model Verification ===" << std::endl;
    
    // Grid Setup: 10x10x10x3
    int nm = 10, na = 10, nh = 10, nz = 3;
    
    std::vector<double> m_nodes(nm), a_nodes(na), h_nodes(nh);
    for(int i = 0; i < nm; ++i) m_nodes[i] = 20.0 * std::pow((double)i/(nm-1), 2);
    for(int i = 0; i < na; ++i) a_nodes[i] = 30.0 * i / (na - 1);
    for(int i = 0; i < nh; ++i) h_nodes[i] = 40.0 * i / (nh - 1);
    
    UnifiedGrid m_g(m_nodes), a_g(a_nodes), h_g(h_nodes);
    Monad::MultiDimGrid grid(m_g, a_g, h_g, nz);
    
    std::cout << "Grid: " << grid.N_m << "x" << grid.N_a << "x" << grid.N_h << "x" << grid.N_z 
              << " = " << grid.total_size << std::endl;
    
    Monad::ThreeAssetParam params;
    params.beta = 0.986;
    params.r_m = -0.01;
    params.r_a = 0.02;
    params.r_h = 0.03;
    params.sigma = 2.0;
    params.chi = 0.1;
    params.fiscal.tax_rule.cgt_rate = 0.15;

    auto process = Monad::IncomeProcessFactory::make_rouwenhorst(0.966, 0.5, nz);
    IncomeProcess income;
    income.n_z = nz;
    income.z_grid = process.z_grid;
    income.Pi_flat = process.Pi_flat;

    auto cpu = std::make_unique<Monad::CpuBackend>(grid, params);
    Monad::ThreeAssetSolver solver(std::move(cpu));

#ifdef MONAD_GPU
    Monad::GpuBackend gpu(nm, na, nh, nz);
    gpu.upload_grids(m_nodes.data(), a_nodes.data(), h_nodes.data(), income.z_grid.data(), 
                     income.Pi_flat.data(), nm, na, nh, nz);
    gpu.set_params(params.beta, params.r_m, params.r_a, params.r_h, params.chi, params.sigma,
                   params.fiscal.tax_rule.lambda, params.fiscal.tax_rule.tau, 
                   params.fiscal.tax_rule.transfer, params.fiscal.tax_rule.cgt_rate);
#endif
    
    Monad::ThreeAssetPolicy cpu_pol(grid.total_size), gpu_pol(grid.total_size);
    Monad::ThreeAssetPolicy cpu_next(grid.total_size), gpu_next(grid.total_size);
    
    // Initial Guess
    for(int flat = 0; flat < grid.total_size; ++flat) {
        int im, ia, ih, iz;
        grid.get_coords(flat, im, ia, ih, iz);
        double m = m_nodes[im], a = a_nodes[ia], h = h_nodes[ih], z = income.z_grid[iz];
        double inc = params.fiscal.tax_rule.after_tax(z) + 0.02*a + 0.03*h;
        double c = 0.05 * (m + a + h) + inc;
        if(c < 0.1) c = 0.1;
        cpu_pol.c_pol[flat] = c;
        gpu_pol.c_pol[flat] = c;
        cpu_pol.value[flat] = (std::pow(c, 1.0-params.sigma)/(1.0-params.sigma)) / (1.0-params.beta);
        gpu_pol.value[flat] = cpu_pol.value[flat];
    }
    
    std::cout << "\nIter | V_diff (Shadow Mode)" << std::endl;
    std::cout << "-----|---------------------" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 5; ++iter) { // Just a few iterations for verification
        solver.solve_bellman(cpu_pol, cpu_next, income);
        
#ifdef MONAD_GPU
        gpu.solve_bellman_iteration(
            gpu_pol.value.data(), gpu_pol.c_pol.data(),
            gpu_next.value.data(), gpu_next.c_pol.data(),
            gpu_next.m_pol.data(), gpu_next.a_pol.data(), gpu_next.h_pol.data(),
            gpu_next.d_a_pol.data(), gpu_next.d_h_pol.data(), gpu_next.adjust_flag.data(),
            grid.total_size);
        
        double max_V_diff = 0.0;
        for(int i = 0; i < grid.total_size; ++i) {
            max_V_diff = std::max(max_V_diff, std::abs(cpu_next.value[i] - gpu_next.value[i]));
        }
        std::cout << std::setw(4) << iter << " | " << std::scientific << std::setprecision(2) << max_V_diff << std::endl;
        
        if (max_V_diff > 1e-12) {
            std::cout << "[ERROR] Precision lost: " << max_V_diff << std::endl;
            return 1;
        }

        std::swap(cpu_pol.value, cpu_next.value);
        std::swap(cpu_pol.c_pol, cpu_next.c_pol);
        std::swap(gpu_pol.value, gpu_next.value);
        std::swap(gpu_pol.c_pol, gpu_next.c_pol);
#endif
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nSUCCESS: Precision maintained across 3 assets." << std::endl;
    std::cout << "Verification time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;

    return 0;
}
