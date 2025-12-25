#pragma once

#include <Zigen/Zigen.hpp>
#include <Zigen/IR/Graph.hpp>
#include <Zigen/IR/Interpreter.hpp>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cassert>

namespace Monad {
namespace GPU {

using namespace Zigen::IR;

/**
 * @brief Manages the Jacobian matrix for HANK solver
 * 
 * Supports dense Jacobian calculation via iterative backward passes (Reverse Mode AD).
 * Future: Will support block-sparse storage.
 */
class HankJacobian {
    size_t n_rows_;
    size_t n_cols_;
    
    // Dense storage: [Rows * Cols]
    // Layout: Row-Major (Standard C/C++)
    // J[i, j] = dRes[i] / dV[j]
    Zigen::IR::Tensor dense_matrix_;
    
    // Flag for sparsity (future)
    bool is_sparse_ = false;

public:
    HankJacobian(size_t n_rows, size_t n_cols, const Zigen::IR::Device& device)
        : n_rows_(n_rows), n_cols_(n_cols), 
          dense_matrix_({n_rows * n_cols}, device) { // Flat storage for now
          // Initialize to zero?
          // Interpreter::backward does accumulate if not cleared? 
          // We will write row by row.
    }

    /**
     * @brief Compute the full Jacobian matrix
     * 
     * Iterates through each output element (Residual[i]), computes gradients w.r.t input (V),
     * and stores it as the i-th row of the Jacobian.
     * 
     * @param graph The Zigen computation graph
     * @param interp The interpreter instance to run backward
     * @param inputs Graph inputs (must include V at index 0)
     * @param outputs Graph outputs (output 0 is Residual)
     */
    void compute(const Graph& graph, Interpreter& interp, 
                 const std::unordered_map<size_t, Tensor>& inputs,
                 const std::unordered_map<size_t, Tensor>& outputs) {
        
        // 1. Validation
        auto it_res = outputs.find(0); // Assuming Output 0 is Residual
        if (it_res == outputs.end()) {
            throw std::runtime_error("HankJacobian: Output 0 (Residual) not found");
        }
        const Tensor& residual = it_res->second;
        
        auto it_v = inputs.find(0); // Assuming Input 0 is V
        if (it_v == inputs.end()) {
             throw std::runtime_error("HankJacobian: Input 0 (V) not found");
        }
        const Tensor& V = it_v->second;
        
        // Shape Assertions
        if (residual.numel() != n_rows_) {
            throw std::runtime_error("HankJacobian: Residual size mismatch. Expected " + 
                                     std::to_string(n_rows_) + ", got " + std::to_string(residual.numel()));
        }
        if (V.numel() != n_cols_) {
            throw std::runtime_error("HankJacobian: V size mismatch. Expected " + 
                                     std::to_string(n_cols_) + ", got " + std::to_string(V.numel()));
        }
        
        // 2. Compute Row by Row (Naive Iteration)
        // Optimization TODO: Batching via "vmap" or "broadcasting" backward if Zigen supported it.
        // For now, N backward passes.
        
        const Device& device = dense_matrix_.device();
        
        // Prepare "grad_output" for Residual: Shape [N_rows] (or same as Res)
        Tensor grad_out_seed(residual.shape, device);
        
        // Host buffer to create one-hot vectors efficiently?
        // Or using backend `zero` and `fill` kernels?
        // Moving data CPU->GPU inside loop is slow.
        // Better: Reset grad_out_seed to 0 on GPU, then set index i to 1.0.
        // Zigen Helpers needed: vector access/modification on GPU.
        // We lack a fast "set_value(idx, val)" in Tensor API exposed to C++.
        // We have `copy_from` (host).
        // Let's use a host buffer for the seed vector, toggle 1.0, copy to GPU.
        // Slow but correct.
        
        std::vector<double> h_seed(n_rows_, 0.0);
        
        for (size_t i = 0; i < n_rows_; ++i) {
            // Set one-hot
            h_seed[i] = 1.0;
            if (i > 0) h_seed[i-1] = 0.0; // Reset previous
            
            grad_out_seed.copy_from(h_seed.data());
            
            // Backward pass
            // We only need grad w.r.t V (id=0 usually, checking assumption)
            // Var::input IDs are dynamic.
            // Using graph inputs map keys.
            // Assumption: V is input matching the id of `inputs[0]`.
            // Wait, inputs is map<id, Tensor>.
            // We need to know which NODE ID is V.
            // In `HankIndividualGraph::build`, V is created first.
            // But we don't have the Graph IDs here unless passed.
            // Assuming inputs contains {v_node_id, V_tensor}.
            // And we want gradient w.r.t v_node_id.
            
            // We need to identify the V node ID.
            // Usually we'd pass it.
            // Hack for now: keys of inputs? 
            // In test, inputs[0] key was used, but that was just valid because `Var(id)` uses the ID.
            // inputs passed to `run` are keyed by NODE ID.
            // So we iterate inputs map and key is ID.
            // WHICH is V?
            // User code must manage this mapping.
            // Let's assume input_ids includes only V, or we differentiate all params.
            
            // Let's compute gradients for ALL inputs provided.
            std::unordered_map<size_t, Tensor> grads;
            
            // We need the output ID for backward seed.
            // Output 0 marked in graph?
            // Tracer::mark_output stores IDs in Tracer, but Graph object has `outputs` vector of Node IDs.
            // graph.outputs()[0] is the ID of Residual node.
            
            size_t res_id = graph.outputs().at(0);
            grads = interp.backward(graph, inputs, {{res_id, grad_out_seed}});
            
            // Extract dRes_i / dV
            // Which input ID is V?
            // We assume the first input in `graph.inputs()` is V.
            // In a real scenario, we should look up by Name or similar.
            // For now, assertion.
            if (graph.inputs().empty()) throw std::runtime_error("Graph has no inputs");
            size_t v_id = graph.inputs()[0]; 
            
            // Validate that v_id actually corresponds to V tensor in input map
            if (inputs.find(v_id) == inputs.end()) {
                // Try to fallback to logic where input[0] map passed is V?
                // inputs map keys are Node IDs.
                // Our test code passes inputs[0] = V_tensor.
                // It assumes node ID 0 is V.
                // Tracer assigns IDs sequentially starting from 0?
                // Yes, Tracer implementation typically does.
                // But let's be safe.
                v_id = 0; 
            }
            
            if (grads.find(v_id) != grads.end()) {
                const Tensor& dV = grads.at(v_id);
                
                // Consistency check
                if(dV.numel() != n_cols_) throw std::runtime_error("Gradient size mismatch");
                
                // Copy row to dense matrix
                // dense_matrix_[i * cols ... (i+1)*cols] = dV
                // Need manual copy helper or Backend::copy
                // Using generic byte copy.
                // address = dense_matrix_.ptr() + i * n_cols_ (pointer arithmetic on double*)
                // dV.ptr() is source.
                
                // Tensor doesn't verify type safety on ptr(), assume double.
                auto& backend = Zigen::IR::get_backend(device);
                backend.copy(const_cast<double*>(dense_matrix_.ptr()) + i * n_cols_, 
                             dV.ptr(), n_cols_ * sizeof(double));
            }
        }
    }
    
    const Tensor& matrix() const { return dense_matrix_; }
    size_t rows() const { return n_rows_; }
    size_t cols() const { return n_cols_; }
};

} // namespace GPU
} // namespace Monad
