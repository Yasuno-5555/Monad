#pragma once

#include <Zigen/Zigen.hpp>
#include <Zigen/IR/Graph.hpp>
#include <Zigen/IR/Interpreter.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "HankIndividualGraph.hpp"
#include "HankJacobian.hpp"

namespace Monad {
namespace GPU {

using namespace Zigen::IR;

/**
 * @brief Hybrid GPU-CPU Newton Solver for HANK models
 * 
 * Orchestrates:
 * 1. GPU Graph Execution (Zigen) -> Residual
 * 2. GPU Jacobian Construction (HankJacobian) -> J
 * 3. CPU Linear Solve (Eigen) -> Delta V
 * 4. GPU State Update -> V_new
 * 
 * Logic strictly separates numerical heavy lifting (Zigen/Eigen) from orchestration.
 */
class MonadHybridSolver {
public:
    struct Params {
        HankIndividualGraph::Params hank_params;
        size_t max_iter = 100;
        double tol = 1e-8;
        double damping = 1.0; // Future use
    };

private:
    Params params_;
    Graph graph_;
    Interpreter interpreter_;
    std::unique_ptr<HankJacobian> jacobian_;
    Device device_;
    
    // State
    Tensor V_;
    // Inputs Map for Graph
    std::unordered_map<size_t, Tensor> inputs_;
    
    // Aux inputs (r, w, etc.) - stored to keep them alive if needed
    Tensor r_tensor_;
    Tensor w_tensor_;
    Tensor a_grid_tensor_;
    Tensor z_grid_tensor_;

public:
    MonadHybridSolver(const Params& p, const Device& device)
        : params_(p), device_(device) {
        
        // 1. Build Graph
        graph_ = HankIndividualGraph::build(params_.hank_params);
        
        // 2. Initialize State V (Guess)
        // Shape [N_z, N_a]
        Shape v_shape = {params_.hank_params.n_z, params_.hank_params.n_a};
        V_ = Tensor(v_shape, device_);
        
        // Initialize V with some guess (e.g. utility of consumption = income)
        // For now, uniform V=0 or similar.
        // Actually, V should be > 0 usually for CRRA Utility? 
        // u(c) = c^(1-gamma)/(1-gamma). If gamma > 1, u < 0. V < 0.
        // Let's safe-init V on host then copy.
        init_initial_guess();
        
        // 3. Initialize other inputs
        // r, w
        r_tensor_ = Tensor({1}, device_);
        w_tensor_ = Tensor({1}, device_);
        // Set values
        std::vector<double> h_r = {p.hank_params.r};
        std::vector<double> h_w = {p.hank_params.w};
        r_tensor_.copy_from(h_r.data());
        w_tensor_.copy_from(h_w.data());
        
        // Grids (a_grid, z_grid)
        // a_grid [1, N_a]
        // z_grid [N_z, 1]
        a_grid_tensor_ = Tensor({1, p.hank_params.n_a}, device_);
        z_grid_tensor_ = Tensor({p.hank_params.n_z, 1}, device_);
        
        init_grids(); // Helper to fill grids
        
        // Bind Inputs using Graph Input IDs
        // Graph inputs list order: V, r, w, a_grid, z_grid
        // We verified this order in graph build?
        // Let's verify graph inputs count.
        if (graph_.inputs().size() != 5) {
            throw std::runtime_error("Graph input count mismatch. Expected 5 (V, r, w, a, z)");
        }
        
        inputs_[graph_.inputs()[0]] = V_;
        inputs_[graph_.inputs()[1]] = r_tensor_;
        inputs_[graph_.inputs()[2]] = w_tensor_;
        inputs_[graph_.inputs()[3]] = a_grid_tensor_;
        inputs_[graph_.inputs()[4]] = z_grid_tensor_;
        
        // 4. Initialize Jacobian
        size_t n_total = params_.hank_params.n_z * params_.hank_params.n_a;
        jacobian_ = std::make_unique<HankJacobian>(n_total, n_total, device_);
    }

    void solve() {
        std::cout << "Starting MonadHybridSolver Newton Loop...\n";
        std::cout << "  Device: " << (device_.type() == DeviceType::CUDA ? "CUDA" : "CPU") << "\n";
        std::cout << "  Grid: " << params_.hank_params.n_a << "x" << params_.hank_params.n_z << "\n";
        
        for (size_t iter = 0; iter < params_.max_iter; ++iter) {
            auto start_iter = std::chrono::high_resolution_clock::now();
            
            // 1. Run Graph (Compute Residual)
            // Inputs are already bound in inputs_ map (V is updated in-place? No, V_ tensor object is same)
            // Note: Tensor assignment V_ = V_new updates strict handle?
            // inputs_ stores copies of Tensor objects (shared storage).
            // So if we update storage of V_, inputs_ sees it.
            
            auto outputs = interpreter_.run(graph_, inputs_);
            
            // Get Residual
            if (outputs.find(graph_.outputs()[0]) == outputs.end()) {
                 throw std::runtime_error("Residual output missing");
            }
            const Tensor& residual = outputs.at(graph_.outputs()[0]);
            
            // 2. Check Convergence (Residual Norm)
            // Copy to CPU for check (and for solve)
            // Optimization: check norm on GPU using Zigen?
            // We need it on CPU for Linear Solve anyway.
            
            size_t N = residual.numel();
            std::vector<double> h_res(N);
            residual.to_cpu(h_res.data());
            
            // Compute Norm L_inf or L2
            double res_norm = 0.0;
            for (double v : h_res) res_norm = std::max(res_norm, std::abs(v));
            
            std::cout << "  Iter " << std::setw(3) << iter 
                      << " | Res Norm: " << std::scientific << std::setprecision(4) << res_norm;
            
            if (res_norm < params_.tol) {
                std::cout << " -> CONVERGED\n";
                return;
            }
            
            // 3. Compute Jacobian
            // Pass outputs map to use the already computed residual if needed (HankJacobian uses generic backward)
            // Though compute() needs the Residual Tensor to verify verification? No.
            // compute() runs backward.
            jacobian_->compute(graph_, interpreter_, inputs_, outputs);
            
            // 4. Linear Solve (CPU Side)
            // Data Transfer: Jacobian (GPU->CPU), Residual (GPU->CPU - already have h_res)
            
            // Jacobian Layout: Row-Major on GPU
            const Tensor& J_gpu = jacobian_->matrix();
            std::vector<double> h_J(J_gpu.numel());
            J_gpu.to_cpu(h_J.data());
            
            // Eigen Map: RowMajor
            // MatrixXd is ColMajor by default.
            // Map<Matrix<double, Dynamic, Dynamic, RowMajor>>
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
                J_eigen(h_J.data(), jacobian_->rows(), jacobian_->cols());
            
            Eigen::Map<Eigen::VectorXd> Res_eigen(h_res.data(), N);
            
            // Solve J * dx = -Res  =>  dx = -J^{-1} * Res
            // Using PartialPivLU
            Eigen::VectorXd dx = J_eigen.partialPivLu().solve(-Res_eigen);
            
            // Delta Norm
            double dx_norm = dx.lpNorm<Eigen::Infinity>();
            std::cout << " | dx Norm: " << dx_norm;
            
            // 5. Update State
            // V_new = V_old + dx
            // Do this on CPU or GPU?
            // V is on GPU.
            // Upload dx to GPU, add? 
            // Or update on CPU and upload entire V?
            // Since we have V on CPU? We didn't download V.
            // Let's download V, update, upload. (Easiest for v1)
            // Or: V_new = V + dx.
            // If we upload dx:
            Tensor dx_tensor(V_.shape, device_);
            dx_tensor.copy_from(dx.data());
            
            // Zigen Add?
            // We can use interpreter to run "Add" graph?
            // Or simple add kernel if exposed?
            // Tensor doesn't have operator+ exposed.
            // Backend has `add`.
            auto& backend = Zigen::IR::get_backend(device_);
            // In-place update V_ = V_ + dx_
            // backend.add(v, dx, v, n)
            
            // Hack with simple Backend call
            // Safe? Storage pointers accessible?
            // Tensor.ptr() is const... wait.
            // Tensor is copyable handle. Storage is shared.
            // We need a way to mutate V_.
            // Zigen::IR::Var is immutable-ish logic.
            // But Tensor (runtime) is mutable storage.
            // Tensor class has no "mutable_ptr()".
            // We can re-assign inputs_[0] to a NEW Tensor?
            // Yes.
            
            // Option A: V_new = V_old + dx via CPU
            // Download V check? No, we trust GPU state.
            // Let's use CPU update for simplicity and safety against Zigen const-correctness.
            std::vector<double> h_V(N);
            V_.to_cpu(h_V.data());
            for(size_t i=0; i<N; ++i) h_V[i] += dx[i];
            
            // Upload
            V_.copy_from(h_V.data()); // V_ is kept alive in valid inputs_
            
            auto end_iter = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end_iter - start_iter;
            std::cout << " | Time: " << std::fixed << std::setprecision(1) << elapsed.count() << "ms\n";
        }
        
        std::cout << "WARNING: Max iterations reached without convergence.\n";
    }

    const Tensor& get_V() const { return V_; }

private:
    void init_initial_guess() {
        // Simple guess: V(z, a) = a?
        // Or something more HANK-like.
        // Utility of r*a + w*z?
        size_t N = utils_total_size();
        std::vector<double> h_data(N);
        // Fill with dummy
        for(size_t i=0; i<N; ++i) h_data[i] = -1.0; // Negative utility
        V_.copy_from(h_data.data());
    }
    
    void init_grids() {
        // a_grid linear [a_min, a_max]
        size_t n_a = params_.hank_params.n_a;
        double da = (params_.hank_params.a_max - params_.hank_params.a_min) / (n_a - 1);
        std::vector<double> h_a(n_a);
        for(size_t i=0; i<n_a; ++i) h_a[i] = params_.hank_params.a_min + i * da;
        a_grid_tensor_.copy_from(h_a.data());
        
        // z_grid dummy [1.0, 1.0...] for now
        size_t n_z = params_.hank_params.n_z;
        std::vector<double> h_z(n_z, 1.0);
        z_grid_tensor_.copy_from(h_z.data());
    }
    
    size_t utils_total_size() const {
        return params_.hank_params.n_a * params_.hank_params.n_z;
    }
};

} // namespace GPU
} // namespace Monad
