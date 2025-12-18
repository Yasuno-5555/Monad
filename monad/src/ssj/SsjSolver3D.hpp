#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include "SparseMatrixBuilder.hpp"
#include "JacobianBuilder3D.hpp"
#include "FakeNewsAggregator.hpp"
#include "../grid/MultiDimGrid.hpp"
#include "../Params.hpp"

namespace Monad {

class SsjSolver3D {
    const MultiDimGrid& grid;
    const TwoAssetParam& params;
    const IncomeProcess& income;

    // Steady State Objects
    TwoAssetPolicy pol_ss;
    std::vector<double> D_ss;
    SparseMat Lambda_ss;

    // Components
    JacobianBuilder3D jac_builder;
    std::unique_ptr<FakeNewsAggregator> fn_agg;

public:
    SsjSolver3D(const MultiDimGrid& g, const TwoAssetParam& p, const IncomeProcess& inc,
                const TwoAssetPolicy& p_ss, const std::vector<double>& d_ss,
                const std::vector<double>& evm, const std::vector<double>& ev)
        : grid(g), params(p), income(inc), pol_ss(p_ss), D_ss(d_ss),
          jac_builder(g, p, inc, evm, ev) // Initialize JB3D
    {
        // 1. Build Sparse Transition Matrix (Lambda)
        SparseMatrixBuilder sp_builder(g, inc);
        Lambda_ss = sp_builder.build_transition_matrix(pol_ss);
        
        // 2. Initialize Fake News Aggregator
        fn_agg = std::make_unique<FakeNewsAggregator>(g, inc, Lambda_ss);
    }

    // Compute the General Equilibrium Jacobian Matrix "H" or specific block Jacobians
    // Returns map: OutputVar -> InputVar -> Matrix (T x T)
    std::map<std::string, std::map<std::string, Eigen::MatrixXd>> compute_block_jacobians(int T) {
        
        // 1. Compute Policy Derivatives (∂y/∂U) at t=0...T
        auto partials = jac_builder.compute_partials(pol_ss); 

        // 2. Compute Distribution Response (Fake News)
        std::map<std::string, std::map<std::string, Eigen::MatrixXd>> J;

        for (auto const& [input_name, pol_map] : partials["m"]) { // Iterate inputs present in m-policy
            
            // Get policy derivatives vectors
            const auto& dm_du = partials["m"][input_name];
            const auto& da_du = partials["a"][input_name]; // Illiquid
            const auto& dc_du = partials["c"][input_name]; // Consumption
            
            // A. Compute Fake News Vector F for distribution
            std::vector<double> F = fn_agg->compute_fake_news_vector(dm_du, da_du, pol_ss, D_ss);
            
            // B. Build Jacobian for Aggregates (e.g. Aggregate Liquid B)
            J["B"][input_name] = build_J_matrix_state(T, F, 0); // 0 for Liquid (m)
            J["K"][input_name] = build_J_matrix_state(T, F, 1); // 1 for Illiquid (a)
            
            // C. Consumption Jacobian
            J["C"][input_name] = build_J_matrix_variable(T, F, dc_du);
        }

        return J;
    }

private:
    // Helper: Build J matrix for a state variable (type 0: m, 1: a)
    // Output = sum(state * dD)
    Eigen::MatrixXd build_J_matrix_state(int T, std::vector<double> F, int type) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(T, T);
        
        // Pre-compute state vector for dot product
        Eigen::VectorXd X(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) {
            int im, ia, iz;
            grid.get_coords(i, im, ia, iz);
            if(type == 0) X[i] = grid.m_grid.nodes[im];
            else          X[i] = grid.a_grid.nodes[ia];
        }

        // Toeplitz Construction via Impulse Response
        std::vector<double> response(T);
        Eigen::VectorXd current = Eigen::Map<Eigen::VectorXd>(F.data(), F.size());
        
        for(int k=0; k<T; ++k) {
            response[k] = X.dot(current);
            current = Lambda_ss * current; // Sparse-Vector product
        }
        
        // Fill Toeplitz
        for(int t=0; t<T; ++t) {
            for(int s=0; s<=t; ++s) {
                J(t, s) = response[t-s];
            }
        }
        
        return J;
    }
    
    // Helper for Consumption (Policy changes + Dist changes)
    Eigen::MatrixXd build_J_matrix_variable(int T, std::vector<double> F, const std::vector<double>& dc_du) {
        // dC_agg[t] = sum( C_ss * dD[t] ) + sum( dC[t] * D_ss )
        
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(T, T);
        
        // 1. Distribution Effect (Off-Diagonal / Persistence)
        Eigen::VectorXd C_ss_vec(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) C_ss_vec[i] = pol_ss.c_pol[i];
        
        std::vector<double> dist_response(T);
        Eigen::VectorXd current = Eigen::Map<Eigen::VectorXd>(F.data(), F.size());
        
        for(int k=0; k<T; ++k) {
            dist_response[k] = C_ss_vec.dot(current);
            current = Lambda_ss * current;
        }
        
        for(int t=0; t<T; ++t) {
            for(int s=0; s<=t; ++s) {
                J(t, s) = dist_response[t-s];
            }
        }
        
        // 2. Direct Policy Effect (Diagonal)
        // dC_direct[t] = sum( dc_du * D_ss ) * shock[t]
        double direct_impact = 0.0;
        for(int i=0; i<grid.total_size; ++i) {
            direct_impact += dc_du[i] * D_ss[i];
        }
        
        for(int t=0; t<T; ++t) {
            J(t, t) += direct_impact;
        }
        
        return J;
    }
};

} // namespace Monad
