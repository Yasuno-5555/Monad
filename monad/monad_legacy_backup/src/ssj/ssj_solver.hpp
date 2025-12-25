#pragma once
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "jacobian_builder.hpp"
#include "aggregator.hpp"
#include "../blocks/SimpleBlocks.hpp"
#include "../blocks/FiscalBlock.hpp"
#include "../blocks/WageBlock.hpp"

namespace Monad {

class SsjSolver {
public:
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;

    // Solve for New Keynesian Transition (MP Shock)
    // Unknown sequence: Output Gap (dY)
    // Market Clearing: Y = C(r, Y)
    // Residual: H * dY = -dShock
    static Vec solve_nk_transition(
        int T,
        const UnifiedGrid& grid,
        const Vec& D_ss,
        const Eigen::SparseMatrix<double>& Lambda_ss,
        const std::vector<double>& a_ss,
        const std::vector<double>& c_ss,
        const JacobianBuilder::PolicyPartials& partials,
        const IncomeProcess& income,
        double r_ss // Added r_ss for Fiscal Block
    ) {
        std::cout << "[Monad::SSJ] Building NK General Equilibrium Jacobian..." << std::endl;

        // --- 1. Block Jacobians ---
        double beta = 0.985;
        double kappa = 0.1;
        double phi_pi = 1.5;
        
        // Fiscal Params
        double B_ss = 0.5; // Debt = 50% GDP
        double phi_B = 0.1; // Tax Rule Sensitivity (Speed of repayment - can be lower now)
        double psi_B = 1.0; // Immediate Pass-through (Stabilizer)

        Mat J_pi_Y = SimpleBlocks::build_nkpc_jacobian(T, beta, kappa);
        Mat J_i_pi = SimpleBlocks::build_taylor_jacobian(T, phi_pi);
        auto J_fisher = SimpleBlocks::build_fisher_jacobians(T);
        
        Mat M_r_Y = (J_fisher.J_r_i * J_i_pi + J_fisher.J_r_pi) * J_pi_Y;

        // Fiscal Matrix: dT = M_T_r * dr
        Mat M_T_r = FiscalBlock::build_tax_response_matrix(T, r_ss, B_ss, phi_B, psi_B);

        // --- 2. Household Jacobians ---
        Eigen::Map<const Vec> c_ss_vec(c_ss.data(), c_ss.size());

        Vec irf_C_r = JacobianAggregator::build_consumption_impulse_response(
            T, grid, D_ss, Lambda_ss, a_ss, c_ss_vec, 
            partials.da_dr, partials.dc_dr, income
        );
        Mat J_C_r = Mat::Zero(T, T);
        for(int c=0; c<T; ++c) for(int r=c; r<T; ++r) J_C_r(r, c) = irf_C_r(r-c);

        Vec irf_C_w = JacobianAggregator::build_consumption_impulse_response(
            T, grid, D_ss, Lambda_ss, a_ss, c_ss_vec, 
            partials.da_dw, partials.dc_dw, income
        );
        Mat J_C_Y = Mat::Zero(T, T); // This is J_C_Inc
        for(int c=0; c<T; ++c) for(int r=c; r<T; ++r) J_C_Y(r, c) = irf_C_w(r-c);
        
        // Stabilizer: Now handled by Fiscal Rule (psi=1), so no damping needed.
        // J_C_Y = J_C_Y * 1.0; 

        // --- 3. Total GE Jacobian ---
        // Household budget: Inc = Y - T
        // dC = J_C_r * dr + J_C_Y * (dY - dT)
        // Substitute dT = M_T_r * dr:
        // dC = (J_C_r - J_C_Y * M_T_r) * dr + J_C_Y * dY
        // Substitute dr = M_r_Y * dY:
        // dC = [ (J_C_r - J_C_Y * M_T_r) * M_r_Y + J_C_Y ] * dY
        
        Mat Total_J_C_Y = (J_C_r - J_C_Y * M_T_r) * M_r_Y + J_C_Y;
        
        // Market Clearing F(Y) = Y - C(Y)
        // dF/dY = I - Total_J_C_Y
        
        Mat H = Mat::Identity(T, T) - Total_J_C_Y;
        
        // Diagnostics
        std::cout << "[Debug] Norm M_T_r: " << M_T_r.norm() << " (Fiscal Feedback)" << std::endl;
        std::cout << "[Debug] Norm H: " << H.norm() << std::endl;

        // --- 4. Shock: Monetary Policy Shock (epsilon_i) ---
        Vec d_eps = Vec::Zero(T);
        d_eps(0) = 0.0025; // 25bp shock (annual 1%)
        double rho = 0.8;
        for(int t=1; t<T; ++t) d_eps(t) = 0.8 * d_eps(t-1);
        
        Vec dr_shock = J_fisher.J_r_i * d_eps; // Direct effect on real rate
        Vec dC_shock = J_C_r * dr_shock;       // Direct effect on Consumption
        
        std::cout << "[Debug] Norm dC_shock: " << dC_shock.norm() << std::endl;
        std::cout << "[Debug] dC_shock(0): " << dC_shock(0) << std::endl;
        
        // Solve H * dY = dC_shock (Expected Consumption drop)
        std::cout << "[Monad::SSJ] Solving NK GE System..." << std::endl;
        Vec dY = H.partialPivLu().solve(dC_shock); 
        
        return dY;
    }

    // v1.8: Struct to hold full macro transition path
    struct TransitionResult {
        Vec dr;  // Output Gap (Y) - naming kept for compatibility or change to dY? Let's use dY.
        Vec dY;
        Vec dC;
        Vec dN;
        Vec dw;
        Vec dpi;
        Vec di;
        Vec dreal_r; // Real rate
        Vec dB;
        Vec dT;
    };

    // v1.7: NK Transition with STICKY WAGES
    // Extended model adds wage Phillips curve channel
    // Unknown sequences: dY (output), dw (wage)
    static TransitionResult solve_nk_wage_transition(
        int T,
        const UnifiedGrid& grid,
        const Vec& D_ss,
        const Eigen::SparseMatrix<double>& Lambda_ss,
        const std::vector<double>& a_ss,
        const std::vector<double>& c_ss,
        const JacobianBuilder::PolicyPartials& partials,
        const IncomeProcess& income,
        double r_ss,
        double theta_w = 0.75, // Wage stickiness parameter
        double phi_pi = 1.5,   // Taylor Rule coefficient (v1.7 tuning)
        double alpha = 0.33    // Capital share
    ) {
        std::cout << "[Monad::SSJ v1.8] Building NK+Wage GE Jacobian..." << std::endl;

        // --- Parameters ---
        double beta = 0.985;
        double kappa_p = 0.1;    // Price Phillips curve slope
        
        // Wage Block Parameters
        WageBlock::Params wage_params;
        wage_params.beta = beta;
        wage_params.theta_w = theta_w;
        wage_params.sigma = 2.0;
        wage_params.phi = 1.0;
        wage_params.compute_kappa();
        
        std::cout << "[Wage] theta_w = " << theta_w << ", kappa_w = " << wage_params.kappa_w << std::endl;
        std::cout << "[Policy] phi_pi = " << phi_pi << std::endl;

        // --- 1. Standard NK Block Jacobians ---
        Mat J_pi_Y = SimpleBlocks::build_nkpc_jacobian(T, beta, kappa_p);
        Mat J_i_pi = SimpleBlocks::build_taylor_jacobian(T, phi_pi);
        auto J_fisher = SimpleBlocks::build_fisher_jacobians(T);
        Mat M_r_Y = (J_fisher.J_r_i * J_i_pi + J_fisher.J_r_pi) * J_pi_Y;

        // --- 2. Wage Block Jacobians ---
        auto wage_jacs = WageBlock::build_all_jacobians(T, wage_params, alpha);
        
        // Goods market with wage channel:
        // Labor Income Inc = w * N
        // dInc = dw + dN
        // dN ~ dY (Production function N ~ Y approx)
        // dC = J_C_r * dr + J_C_inc * dInc
        //    = J_C_r * dr + J_C_w * (dw + dY)  (Using J_C_w as proxy for J_C_inc)
        
        // Substitution:
        // dr = M_r_Y * dY
        // dw = J_w_Y * dY
        
        // --- 3. Household Jacobians ---
        Eigen::Map<const Vec> c_ss_vec(c_ss.data(), c_ss.size());

        Vec irf_C_r = JacobianAggregator::build_consumption_impulse_response(
            T, grid, D_ss, Lambda_ss, a_ss, c_ss_vec, 
            partials.da_dr, partials.dc_dr, income
        );
        Mat J_C_r = Mat::Zero(T, T);
        for(int c=0; c<T; ++c) for(int r=c; r<T; ++r) J_C_r(r, c) = irf_C_r(r-c);

        Vec irf_C_w = JacobianAggregator::build_consumption_impulse_response(
            T, grid, D_ss, Lambda_ss, a_ss, c_ss_vec, 
            partials.da_dw, partials.dc_dw, income
        );
        Mat J_C_w = Mat::Zero(T, T);
        for(int c=0; c<T; ++c) for(int r=c; r<T; ++r) J_C_w(r, c) = irf_C_w(r-c);

        // --- 4. Construct Full GE System ---
        Mat J_w_Y = wage_jacs.J_dw_mrs * (wage_params.sigma * J_C_w + 
                                          wage_params.phi * Mat::Identity(T, T));
        
        // dC = [ J_C_r * M_r_Y + J_C_w * (J_w_Y + I) ] * dY
        Mat Total_J_C_Y = J_C_r * M_r_Y + J_C_w * (J_w_Y + Mat::Identity(T, T));
        
        // Market Clearing: dY - dC = 0
        Mat H = Mat::Identity(T, T) - Total_J_C_Y;

        // --- 5. Shock ---
        Vec d_eps = Vec::Zero(T);
        d_eps(0) = 0.0025;  // 25bp shock
        for(int t=1; t<T; ++t) d_eps(t) = 0.8 * d_eps(t-1);
        
        Vec dr_shock = J_fisher.J_r_i * d_eps;
        Vec dC_shock = J_C_r * dr_shock;
        
        std::cout << "[Monad::SSJ v1.8] Solving NK+Wage GE System..." << std::endl;
        Vec dY = H.partialPivLu().solve(dC_shock);
        
        // --- 6. Recover Full Paths ---
        Vec dw = J_w_Y * dY;
        Vec dpi = J_pi_Y * dY;
        Vec di = J_i_pi * dpi + d_eps; // Include shock in nominal rate
        // dr = di - dpi(+1)
        // Or dr = M_r_Y * dY + shock
        Vec dreal_r = M_r_Y * dY + dr_shock; 
        
        Vec dC = Total_J_C_Y * dY + dC_shock; // Verification: should equal dY
        
        Vec dN = dY; // Production function approx
        
        // Fiscal variables (Simplified: No B feedback yet in matrix, but can compute ex-post)
        // dT = T_ss * (tau * dY) ? No, tau is rate.
        // Assuming T = tau * Y (income tax) -> dT = tau * dY
        Vec dT = dY * 0.15; // Approx tau=0.15
        
        // B dynamics: B_t = (1+r)B_{t-1} + G - T
        // dB_t = (1+r)dB_{t-1} + B_{t-1}*dr_t - dT_t
        Vec dB = Vec::Zero(T);
        double B_ss_val = 0.5; // 50% GDP
        for(int t=0; t<T; ++t) {
            double prev_dB = (t==0) ? 0.0 : dB(t-1);
            dB(t) = (1.0 + r_ss) * prev_dB + B_ss_val * dreal_r(t) - dT(t);
        }

        std::cout << "[Result] Peak dw: " << dw.head(10).maxCoeff() 
                  << " Peak dY: " << dY.minCoeff() << std::endl;
        
        return {dY, dY, dC, dN, dw, dpi, di, dreal_r, dB, dT}; // dr=dY for compatibility? No, explicit fields.
    }
};

} // namespace Monad

