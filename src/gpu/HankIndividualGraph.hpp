#pragma once

#include <Zigen/Zigen.hpp>
#include <Zigen/Zigen.hpp>
#include <Zigen/IR/Graph.hpp>
#include <Zigen/IR/Tracer.hpp>
#include <Zigen/IR/Var.hpp>
#include <vector>
#include <string>

namespace Monad {
namespace GPU {

using namespace Zigen::IR;

/**
 * @brief Zigen Graph definition for HANK Individual Block
 * 
 * Implements the Bellman Update / Euler Residual logic using Zigen Ops.
 * Uses Upwind Finite Difference explicitly.
 */
class HankIndividualGraph {
public:
    struct Params {
        size_t n_a;
        size_t n_z;
        double a_min;
        double a_max;
        double gamma; // Risk aversion
        double rho;   // Time preference
        double r;     // Interest rate
        double w;     // Wage
        // Grid can be non-uniform, but for v1 we use bounds + N
    };

    /**
     * @brief Build the computation graph
     * 
     * Inputs:
     *   0: V (Value Function) [N_z, N_a]
     *   1: r (Interest Rate) [Scalar] - Optional if treated as param input
     *   2: w (Wage) [Scalar]         - Optional if treated as param input
     * 
     * Outputs:
     *   0: Residual (V_new - V_old) or Bellman Error
     */
    static Graph build(const Params& p) {
        Tracer::instance().begin();

        // --- Inputs ---
        // Shape: [N_z, N_a]
        Shape v_shape = {p.n_z, p.n_a};
        
        auto V = Var::input(v_shape, DType::Float64, "V");
        auto r = Var::input({1}, DType::Float64, "r"); // Dynamic parameter
        auto w = Var::input({1}, DType::Float64, "w"); // Dynamic parameter
        
        // --- Constants ---
        double da = (p.a_max - p.a_min) / (p.n_a - 1);
        auto c_gamma = Var::constant(p.gamma);
        auto c_rho = Var::constant(p.rho);
        auto c_da = Var::constant(da);
        auto c_inv_da = Var::constant(1.0 / da);
        
        // Construct Grid tensors (constant for now)
        // a_grid: [1, N_a]
        // z_grid: [N_z, 1] (Simplified: placeholder)
        // For simplicity in v1 graph, we assume z and a are handled via
        // broadcasting or simple constructs.
        // To do this "properly" in Zigen, we should pass grid as constant input or build it.
        // Let's assume linear grid for `s = r*a + w*z - c`.
        
        // --- 1. Finite Difference (dV/da) ---
        // V shape: [N_z, N_a]
        // Slices:
        // V_center_left: V[:, 0:-1]
        // V_center_right: V[:, 1:]
        
        // V_left_pad: V[:, 0] (duplicated for boundary) - or handled via Concat?
        // We need:
        // dV_f = (V[plus] - V[cur]) / da
        // dV_b = (V[cur] - V[minus]) / da
        
        // --- Forward Difference ---
        // V_plus: V[:, 1:]
        // V_cur_f: V[:, :-1]
        // dV_f_raw = (V_plus - V_cur_f) * inv_da
        // We need to pad dV_f_raw to size N_a. At right boundary, use backward value (or linear extrapolation).
        
        auto V_plus = V.slice({0, 1}, {p.n_z, p.n_a - 1});
        auto V_cur_f = V.slice({0, 0}, {p.n_z, p.n_a - 1});
        auto dV_f_raw = (V_plus - V_cur_f) * c_inv_da;
        
        // Pad right boundary with last column of dV_f_raw
        // dV_f_last = dV_f_raw[:, -1] -> Slice last col, shape [N_z, 1]
        auto dV_f_last = dV_f_raw.slice({0, p.n_a - 2}, {p.n_z, 1});
        // Note: p.n_a - 2 is the index of the last element in array of size p.n_a - 1
        
        // Concatenate: [dV_f_raw, dV_f_last]
        // BUT slicing in Zigen cuts dim 1? No, we said axis=0 for Concat.
        // Wait, our Concat implementation is HARDCODED for axis=0 (batch dim).
        // HANK usually has (z, a) or (a, z).
        // My inputs are [N_z, N_a].
        // Concat along a (dim 1) is needed here!
        // My previous Concat implementation works on OUTER dim (axis=0).
        // If I want to concat along dim 1, I need to Transpose, Concat, Transpose back.
        // Or implement axis support in Concat.
        
        // Let's use Transpose strategy for now as it uses existing ops.
        
        // Transpose V to [N_a, N_z] for easier processing along 'a' (now axis 0)
        auto V_T = transpose(V); // [N_a, N_z]
        
        // Re-slice on Transposed
        auto V_T_plus = V_T.slice({1, 0}, {p.n_a - 1, p.n_z}); // [1:, :]
        auto V_T_cur_f = V_T.slice({0, 0}, {p.n_a - 1, p.n_z}); // [:-1, :]
        auto dV_f_raw_T = (V_T_plus - V_T_cur_f) * c_inv_da; // [N_a-1, N_z]
        
        // Last row: dV_f_raw_T[-1, :]
        auto dV_f_last_T = dV_f_raw_T.slice({p.n_a - 2, 0}, {1, p.n_z});
        
        // Concat along axis 0
        auto dV_f_T = Var::concat({dV_f_raw_T, dV_f_last_T}, 0); // [N_a, N_z]
        auto dV_f = transpose(dV_f_T); // [N_z, N_a]
        
        // --- Backward Difference ---
        // dV_b_raw = (V[1:] - V[:-1]) / da -- same as f_raw, but shifted position
        // dV_b needs to start with Forward difference at left boundary? Or boundary condition?
        // Usually State Constraint: s >= 0. Consumption can't be negative?
        // At a_min, we cannot save less, so backward difference is valid?
        // Usually dV_b[0] = dV_f[0] or standard B.C.
        // Let's use dV_f[0] (linear extrapolation) for left boundary of dV_b.
        
        auto dV_b_first_T = dV_f_raw_T.slice({0, 0}, {1, p.n_z});
        auto dV_b_T = Var::concat({dV_b_first_T, dV_f_raw_T}, 0);
        auto dV_b = transpose(dV_b_T);
        
        // --- Upwind Selection ---
        // c = (u')^-1(dV)
        // u(c) = c^(1-gamma)/(1-gamma) -> u'(c) = c^-gamma -> c = (dV)^(-1/gamma)
        // V is concave, dV decreases.
        // drift s = r*a + w*z - c
        // if s > 0 use dV_f, if s < 0 use dV_b.
        
        // Calculate c_f and c_b
        auto c_f = pow(dV_f, -1.0 / p.gamma);
        auto c_b = pow(dV_b, -1.0 / p.gamma); // Note: dV must be positive.
        // If dV < 0 (numerical error?), we might need ReLU or epsilon.
        // Zigen `pow` supports element-wise.
        
        // Compute drift
        // We need 'a' grid and 'z' grid.
        // a grid is constant row vector [1, N_a]
        // z grid is constant col vector [N_z, 1]
        // Currently constructing them via inputs or constants is tricky without `arange`.
        // Let's assume passed as inputs 3 and 4 for now to proceed fast?
        // Or constructing constant tensor in code.
        
        // For Upwind, we select c based on drift s.
        // s = r*a + w*z - c
        // Note: 'a' and 'z' are needed as tensors.
        // We will assume they are passed as inputs 3 and 4 for this graph version.
        // Input 3: a_grid [1, N_a] (broadcasts against N_z)
        // Input 4: z_grid [N_z, 1] (broadcasts against N_a)
        
        auto a_grid = Var::input({1, p.n_a}, DType::Float64, "a_grid");
        auto z_grid = Var::input({p.n_z, 1}, DType::Float64, "z_grid");
        
        // Broadcast r and w
        // r: [1] -> [1, 1] (implicitly handled by scalar ops?) 
        // Var ops handle scalar-tensor and tensor-tensor.
        // r * a_grid -> [1, N_a]
        // w * z_grid -> [N_z, 1]
        // sum -> [N_z, N_a] via broadcast
        
        auto income = r * a_grid + w * z_grid;
        
        auto s_f = income - c_f;
        auto s_b = income - c_b;
        
        // Selection Logic (Upwind)
        // If s_f > 0: use forward
        // If s_b < 0: use backward
        // Else: s=0 approx (or use one of them, usually linear combination or force 0 drift)
        // Standard HANK: 
        // 1. If s_f > 0: forward
        // 2. Else if s_b < 0: backward
        // 3. Else (s_f <= 0 and s_b >= 0): trapped, c = income (s=0)
        
        // We need 'Select' OpCode which was defined but not exposed in Var yet?
        // Wait, OpCode::Select is there. Var::select is NOT there.
        // If s_f > 0: use forward (c_f)
        // Else if s_b < 0: use backward (c_b)
        // Else: trapped, c = income (s=0)
        
        // select(cond, true_val, false_val)
        // cond > 0 -> true_val
        
        auto neg_s_b = -s_b; // neg_s_b > 0 means s_b < 0
        auto c = select(s_f, c_f, select(neg_s_b, c_b, income));
        
        auto s = income - c;
        
        // dV calculation for Residual
        // If we picked c_f, we used dV_f. If c_b, dV_b. If income, drift is 0 so term dV*s is 0.
        // We can just select dV matching the choice of c to be consistent.
        auto dV = select(s_f, dV_f, select(neg_s_b, dV_b, dV_f));
        
        // Bellman Residual: u(c) + dV * s - rho * V
        // Utility u(c) = c^(1-gamma)/(1-gamma)
        auto one_minus_gamma = Var::constant(1.0 - p.gamma);
        auto inv_one_minus_gamma = Var::constant(1.0 / (1.0 - p.gamma));
        
        auto u_c = pow(c, 1.0 - p.gamma) * inv_one_minus_gamma; 
        
        auto residual = u_c + dV * s - c_rho * V;
        
        // Mark Output
        // Output 0: Residual [N_z, N_a]
        residual.mark_output();
        
        return Tracer::instance().end();
    }
};

} // namespace GPU
} // namespace Monad
