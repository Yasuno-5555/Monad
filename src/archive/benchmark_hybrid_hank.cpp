#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>

#include "gpu/HankIndividualGraph.hpp"
#include "gpu/HankJacobian.hpp"
#include "gpu/MonadHybridSolver.hpp"

#include <Zigen/IR/Interpreter.hpp>
#include <Zigen/IR/Device.hpp>

using namespace Zigen::IR;
using namespace Monad::GPU;

struct BenchParams {
    size_t n_z = 7;
    size_t n_a = 500; // Moderate size for CPU to handle quickly
    size_t warmup = 5;
    size_t steps = 20;
};

void run_benchmark(const std::string& device_name, const Device& device, BenchParams bp) {
    std::cout << "\n------------------------------------------------\n";
    std::cout << "Benchmarking on " << device_name << "\n";
    std::cout << "Grid: " << bp.n_z << " x " << bp.n_a << " (" << bp.n_z * bp.n_a << " states)\n";
    std::cout << "------------------------------------------------\n";

    // 1. Setup Data
    HankIndividualGraph::Params p;
    p.n_a = bp.n_a;
    p.n_z = bp.n_z;
    p.a_min = 0.0;
    p.a_max = 50.0;
    p.gamma = 2.0;
    p.rho = 0.05;
    p.r = 0.04;
    p.w = 1.0;

    // Common Inputs
    Shape v_shape = {p.n_z, p.n_a};
    Tensor V(v_shape, device);
    Tensor r({1}, device);
    Tensor w({1}, device);
    Tensor a_grid({1, p.n_a}, device);
    Tensor z_grid({p.n_z, 1}, device);

    // Initialize (Dummy Values)
    std::vector<double> h_v(v_shape.num_elements(), 0.1); // Avoid 0 for log/pow
    V.copy_from(h_v.data());
    
    // 2. Build Graph (Overhead usually once)
    Graph graph = HankIndividualGraph::build(p);
    Interpreter interp;

    // Bind Inputs
    // Assuming Input Order: V, r, w, a, z
    std::unordered_map<size_t, Tensor> inputs;
    if(graph.inputs().size() >= 5) {
        inputs[graph.inputs()[0]] = V;
        inputs[graph.inputs()[1]] = r;
        inputs[graph.inputs()[2]] = w;
        inputs[graph.inputs()[3]] = a_grid;
        inputs[graph.inputs()[4]] = z_grid;
    }

    // --- Res Eval Benchmark ---
    
    // Warmup
    for(size_t i=0; i<bp.warmup; ++i) {
        interp.run(graph, inputs);
    }

    auto start_res = std::chrono::high_resolution_clock::now();
    std::unordered_map<size_t, Tensor> outputs;
    for(size_t i=0; i<bp.steps; ++i) {
        outputs = interp.run(graph, inputs);
    }
    // Block for GPU
    if(device.type() == DeviceType::CUDA) {
        // Force sync if Zigen methods are async (Interpreter run is usually sync in v1, but good practice)
        // Currently no explicit Sync API exposed on Device, relying on Tensor operations blocking copy-back or explicit sync.
        // Assuming Interpreter run is synchronous for now.
    }
    auto end_res = std::chrono::high_resolution_clock::now();
    double time_res = std::chrono::duration<double, std::milli>(end_res - start_res).count() / bp.steps;
    
    std::cout << "Residual Eval Time: " << std::fixed << std::setprecision(3) << time_res << " ms\n";

    // --- Jacobian Build Benchmark ---
    size_t n_total = p.n_z * p.n_a;
    HankJacobian jacobian(n_total, n_total, device);
    
    // Warmup
    for(size_t i=0; i<std::min(bp.warmup, (size_t)2); ++i) { // Jacobian is slow on CPU, less warmup
        jacobian.compute(graph, interp, inputs, outputs);
    }
    
    auto start_jac = std::chrono::high_resolution_clock::now();
    size_t jac_steps = std::min(bp.steps, (size_t)5); // Reduce steps for Jacobian as it's N times slower
    for(size_t i=0; i<jac_steps; ++i) {
        jacobian.compute(graph, interp, inputs, outputs);
    }
    auto end_jac = std::chrono::high_resolution_clock::now();
    double time_jac = std::chrono::duration<double, std::milli>(end_jac - start_jac).count() / jac_steps;
    
    std::cout << "Jacobian Build Time: " << std::fixed << std::setprecision(3) << time_jac << " ms\n";
    
    // --- Full Solver Step Benchmark ---
    // MonadHybridSolver solver(p, device);
    // solver.solve() runs full loop, maybe benchmark 1 iter?
    // We can't tightly loop internal private methods.
    // Just report J + Res as approx step time.
    std::cout << "Approx Newton Step:  " << (time_res + time_jac) << " ms\n";
}

int main() {
    try {
        BenchParams bp;
        bp.n_a = 500;
        bp.n_z = 7;
        
        // 1. CPU Benchmark
        run_benchmark("CPU (Ref)", Device::cpu(), bp);
        
        // 2. GPU Benchmark
        // Check if CUDA available?
        // Zigen::Device::cuda() throws if not available? Or check compile flag?
#ifdef ZIGEN_USE_CUDA
        try {
            run_benchmark("CUDA (Target)", Device::cuda(), bp);
        } catch(const std::exception& e) {
            std::cerr << "CUDA Benchmark Failed: " << e.what() << "\n";
        }
#else
        std::cout << "CUDA Benchmark Skipped (ZIGEN_USE_CUDA not defined)\n";
#endif

    } catch(const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
