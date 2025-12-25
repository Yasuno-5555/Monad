#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>

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
    std::cout << "=== G2.5: Small Grid Convergence (10x10x3) ===" << std::endl;
    
    // SMALL grid for fast convergence
    int nm = 10;
    std::vector<double> m_nodes(nm);
    for(int i = 0; i < nm; ++i) {
        double r = (double)i / (nm - 1);
        m_nodes[i] = 20.0 * r * r;
    }
    UnifiedGrid m_g(m_nodes);
    
    int na = 10;
    std::vector<double> a_nodes(na);
    for(int i = 0; i < na; ++i) {
        a_nodes[i] = 30.0 * i / (na - 1);
    }
    UnifiedGrid a_g(a_nodes);
    
    int nz = 3;
    Monad::MultiDimGrid grid(m_g, a_g, nz);
    std::cout << "Grid: " << grid.N_m << "x" << grid.N_a << "x" << grid.N_z 
              << " = " << grid.total_size << std::endl;
    
    Monad::TwoAssetParam params;
    params.beta = 0.986;
    params.r_m = -0.01;
    params.r_a = 0.02;
    params.sigma = 2.0;
    params.chi = 0.1;

    auto process = Monad::IncomeProcessFactory::make_rouwenhorst(0.966, 0.5, nz);
    IncomeProcess income;
    income.n_z = nz;
    income.z_grid = process.z_grid;
    income.Pi_flat = process.Pi_flat;

    auto cpu = std::make_unique<Monad::CpuBackend>(grid, params);
    Monad::TwoAssetSolver solver(std::move(cpu));

#ifdef MONAD_GPU
    Monad::GpuBackend gpu(grid.N_m, grid.N_a, grid.N_z);
    gpu.upload_grids(m_nodes.data(), a_nodes.data(), income.z_grid.data(), 
                     income.Pi_flat.data(), grid.N_m, grid.N_a, grid.N_z);
    gpu.set_params(params.beta, params.r_m, params.r_a, params.chi, params.sigma,
                   params.fiscal.tax_rule.lambda, params.fiscal.tax_rule.tau, 
                   params.fiscal.tax_rule.transfer);
#endif
    
    Monad::TwoAssetPolicy cpu_pol(grid.total_size), gpu_pol(grid.total_size);
    Monad::TwoAssetPolicy cpu_next(grid.total_size), gpu_next(grid.total_size);
    
    for(int flat = 0; flat < grid.total_size; ++flat) {
        int im, ia, iz;
        grid.get_coords(flat, im, ia, iz);
        double m = grid.m_grid.nodes[im];
        double a = grid.a_grid.nodes[ia];
        double z = income.z_grid[iz];
        double inc = params.r_a * a + params.fiscal.tax_rule.after_tax(z);
        double c = 0.04 * (m + a) + inc;
        if(c < 0.1) c = 0.1;
        cpu_pol.c_pol[flat] = c;
        gpu_pol.c_pol[flat] = c;
        double u = (c <= 0) ? -1e9 : (std::pow(c, 1.0 - params.sigma) / (1.0 - params.sigma));
        cpu_pol.value[flat] = u / (1.0 - params.beta);
        gpu_pol.value[flat] = cpu_pol.value[flat];
    }
    
    const int MAX_ITER = 200;
    const double TOL = 1e-8;
    
    std::cout << "\nIter | dV_cpu     | dV_gpu     | V_diff     | c_diff" << std::endl;
    std::cout << "-----|------------|------------|------------|------------" << std::endl;
    
    bool converged = false;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        solver.solve_bellman(cpu_pol, cpu_next, income);
        
#ifdef MONAD_GPU
        gpu.solve_bellman_iteration(
            gpu_pol.value.data(), gpu_pol.c_pol.data(),
            gpu_next.value.data(), gpu_next.c_pol.data(),
            gpu_next.m_pol.data(), gpu_next.a_pol.data(),
            gpu_next.d_pol.data(), gpu_next.adjust_flag.data(),
            grid.total_size);
        
        double max_dV_cpu = 0.0, max_dV_gpu = 0.0;
        double max_V_diff = 0.0, max_c_diff = 0.0;
        
        for(int i = 0; i < grid.total_size; ++i) {
            max_dV_cpu = std::max(max_dV_cpu, std::abs(cpu_next.value[i] - cpu_pol.value[i]));
            max_dV_gpu = std::max(max_dV_gpu, std::abs(gpu_next.value[i] - gpu_pol.value[i]));
            max_V_diff = std::max(max_V_diff, std::abs(cpu_next.value[i] - gpu_next.value[i]));
            max_c_diff = std::max(max_c_diff, std::abs(cpu_next.c_pol[i] - gpu_next.c_pol[i]));
        }
        
        // Print every 10 iterations or first/last 5
        if (iter < 5 || iter % 20 == 0 || iter > MAX_ITER - 5 || max_dV_cpu < TOL) {
            std::cout << std::setw(4) << iter << " | " 
                      << std::scientific << std::setprecision(2) 
                      << max_dV_cpu << " | " << max_dV_gpu << " | "
                      << max_V_diff << " | " << max_c_diff << std::endl;
        }
        
        std::swap(cpu_pol.value, cpu_next.value);
        std::swap(cpu_pol.c_pol, cpu_next.c_pol);
        std::swap(gpu_pol.value, gpu_next.value);
        std::swap(gpu_pol.c_pol, gpu_next.c_pol);
        
        if (max_dV_cpu < TOL && max_dV_gpu < TOL) {
            std::cout << "\n*** Converged at iteration " << iter << " ***" << std::endl;
            converged = true;
            break;
        }
#else
        std::swap(cpu_pol.value, cpu_next.value);
        std::swap(cpu_pol.c_pol, cpu_next.c_pol);
#endif
    }
    
    if (!converged) {
        std::cout << "\n*** Not converged after " << MAX_ITER << " iterations ***" << std::endl;
    }
    
    // Final comparison
    double final_V_diff = 0.0, final_c_diff = 0.0;
    for(int i = 0; i < grid.total_size; ++i) {
        final_V_diff = std::max(final_V_diff, std::abs(cpu_pol.value[i] - gpu_pol.value[i]));
        final_c_diff = std::max(final_c_diff, std::abs(cpu_pol.c_pol[i] - gpu_pol.c_pol[i]));
    }
    std::cout << "\nFinal: max|V_diff|=" << std::scientific << final_V_diff 
              << " max|c_diff|=" << final_c_diff << std::endl;

    return 0;
}
