#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace Monad {

/**
 * WageBlock: New Keynesian Wage Phillips Curve
 * 
 * Models sticky wages via Calvo pricing:
 *   - Each period, fraction (1 - θw) of workers can reset wages
 *   - Optimal wage setting creates forward-looking wage dynamics
 * 
 * Key equation (linearized NKWPC):
 *   πw_t = β * πw_{t+1} + κw * (mrs_t - w_t)
 * 
 * Where:
 *   πw = wage inflation (dw)
 *   mrs = marginal rate of substitution (disutility of labor)
 *   w = log real wage
 *   κw = (1-θw)(1-β*θw)/θw * slope coefficient
 */
class WageBlock {
public:
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;

    // Wage block parameters
    struct Params {
        double beta = 0.99;      // Discount factor
        double theta_w = 0.75;   // Calvo wage stickiness (avg 4 quarters)
        double sigma = 2.0;      // Risk aversion (for MRS)
        double phi = 1.0;        // Inverse Frisch elasticity
        double kappa_w = 0.0;    // Computed from theta_w if not set
        
        void compute_kappa() {
            // Standard NK formula
            if (kappa_w == 0.0) {
                kappa_w = (1.0 - theta_w) * (1.0 - beta * theta_w) / theta_w;
            }
        }
    };

    /**
     * Build Wage Phillips Curve Jacobian
     * 
     * πw_t = β * πw_{t+1} + κw * gap_t
     * where gap_t = mrs_t - w_t (wage markup gap)
     * 
     * Rearranged: πw_t - β * πw_{t+1} = κw * gap_t
     * Matrix form: A * πw = κw * gap
     * Solution: πw = A^{-1} * κw * gap
     * 
     * Returns J_dw_mrs: Jacobian of wage inflation w.r.t. MRS shocks
     */
    static Mat build_nkwpc_jacobian(int T, const Params& p) {
        Params params = p;
        params.compute_kappa();
        
        // Build (I - β L^{-1}) matrix
        Mat A = Mat::Identity(T, T);
        for (int t = 0; t < T - 1; ++t) {
            A(t, t + 1) = -params.beta;  // Forward expectation
        }
        
        // Jacobian: dπw/dmrs = A^{-1} * κw
        Mat invA = A.inverse();
        return invA * params.kappa_w;
    }

    /**
     * Build MRS (Marginal Rate of Substitution) from consumption and labor
     * 
     * Standard GHH/KPR preferences:
     *   U(C, N) = C^{1-σ}/(1-σ) - χ * N^{1+φ}/(1+φ)
     *   
     * MRS = -U_N / U_C = χ * N^φ * C^σ
     * 
     * Log-linearized around steady state:
     *   mrs_t = σ * c_t + φ * n_t
     * 
     * Returns Jacobian J_mrs_c (impact of consumption on MRS)
     */
    static Mat build_mrs_jacobian_c(int T, double sigma) {
        // mrs = σ * c + φ * n
        // So ∂mrs/∂c = σ
        return Mat::Identity(T, T) * sigma;
    }

    static Mat build_mrs_jacobian_n(int T, double phi) {
        // ∂mrs/∂n = φ
        return Mat::Identity(T, T) * phi;
    }

    /**
     * Labor demand (firm side)
     * 
     * With Cobb-Douglas: w = (1-α) * A * K^α * N^{-α} = MPL
     * Log-linearized: w = mc + mpn
     * 
     * In equilibrium: N adjusts so w = MPL
     * Given sticky w, we solve for N:
     *   n_t = (1/α) * (mpn_ss - w_t + mc_t)
     * 
     * Returns J_n_w: How labor responds to wage changes
     */
    static Mat build_labor_demand_jacobian(int T, double alpha) {
        // dn/dw = -1/α (higher wage → less labor demand)
        return Mat::Identity(T, T) * (-1.0 / alpha);
    }

    /**
     * Combined wage block Jacobians for SSJ integration
     * 
     * Returns struct with all relevant Jacobians
     */
    struct WageJacobians {
        Mat J_dw_mrs;    // Wage inflation response to MRS
        Mat J_mrs_c;     // MRS response to consumption
        Mat J_mrs_n;     // MRS response to labor
        Mat J_n_w;       // Labor demand response to wage
        Mat J_n_r;       // Labor response to interest rate (via K)
    };

    static WageJacobians build_all_jacobians(int T, const Params& p, double alpha) {
        WageJacobians out;
        
        out.J_dw_mrs = build_nkwpc_jacobian(T, p);
        out.J_mrs_c = build_mrs_jacobian_c(T, p.sigma);
        out.J_mrs_n = build_mrs_jacobian_n(T, p.phi);
        out.J_n_w = build_labor_demand_jacobian(T, alpha);
        
        // Labor response to r: via capital-labor ratio
        // K/N ratio rises when r falls → n↑
        // Simplified: dn/dr ≈ -(1-α)/α * (K/Y)
        double KY_ratio = alpha / 0.04;  // Approximate K/Y at 4% interest
        out.J_n_r = Mat::Identity(T, T) * (-(1.0 - alpha) / alpha * KY_ratio * 0.01);
        
        return out;
    }

    /**
     * Compute steady-state MRS gap
     * 
     * In steady state with labor market clearing:
     *   w_ss = mrs_ss (no exploited wage markup in flex-wage equilibrium)
     * 
     * With unemployment, there's a wedge:
     *   mrs_ss < w_ss (workers want to work at current wage but can't)
     */
    static double compute_ss_mrs(double C_ss, double N_ss, double sigma, double phi, double chi = 1.0) {
        // MRS = χ * N^φ * C^σ
        return chi * std::pow(N_ss, phi) * std::pow(C_ss, sigma);
    }
};

} // namespace Monad
