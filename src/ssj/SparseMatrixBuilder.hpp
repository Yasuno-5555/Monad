#pragma once
#include <Eigen/Sparse>
#include <vector>
#include <iostream> 
#include "../grid/MultiDimGrid.hpp"
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::Triplet<double> Triplet;

class SparseMatrixBuilder {
    const MultiDimGrid& grid;
    const IncomeProcess& income;

public:
    SparseMatrixBuilder(const MultiDimGrid& g, const IncomeProcess& inc) 
        : grid(g), income(inc) {}

    // Build the Transition Matrix Lambda (Transpose of iteration matrix)
    // D_{t+1} = Lambda * D_t
    // Note: If T is the operator such that D_{t+1}[j] = sum_i T_{ji} D_t[i],
    // then Lambda_{ji} = Prob(i -> j).
    // In our case, we iterate FROM i TO j.
    // The coefficients are placed in (j, i).
    SparseMat build_transition_matrix(const TwoAssetPolicy& pol) {
        std::vector<Triplet> triplets;
        triplets.reserve(grid.total_size * 10); // Estimate non-zeros

        // Loop over FROM state (im, ia, iz) -> corresponds to Column j (idx_from)
        for (int iz = 0; iz < grid.n_z; ++iz) {
            for (int ia = 0; ia < grid.N_a; ++ia) {
                for (int im = 0; im < grid.N_m; ++im) {
                    
                    int idx_from = grid.idx(im, ia, iz);
                    
                    // 1. Get Policy Choices
                    double m_next = pol.m_pol[idx_from];
                    double a_next = pol.a_pol[idx_from];

                    // 2. Find Weights (Same logic as Aggregator)
                    int im_low, im_high, ia_low, ia_high;
                    double wm_low, wm_high, wa_low, wa_high;
                    
                    find_weights(grid.m_grid, m_next, im_low, im_high, wm_low, wm_high);
                    find_weights(grid.a_grid, a_next, ia_low, ia_high, wa_low, wa_high);

                    // 3. Distribute to TO states
                    int z_offset = iz * grid.n_z;
                    for (int iz_next = 0; iz_next < grid.n_z; ++iz_next) {
                        double prob_z = income.Pi_flat[z_offset + iz_next];
                        if (prob_z < 1e-10) continue;

                        // Target Indices for z_next
                        int idx_LL = grid.idx(im_low, ia_low, iz_next);
                        int idx_HL = grid.idx(im_high, ia_low, iz_next);
                        int idx_LH = grid.idx(im_low, ia_high, iz_next);
                        int idx_HH = grid.idx(im_high, ia_high, iz_next);

                        // Push Triplets (Row=TO, Col=FROM, Value)
                        
                        add_triplet(triplets, idx_LL, idx_from, prob_z * wm_low * wa_low);
                        add_triplet(triplets, idx_HL, idx_from, prob_z * wm_high * wa_low);
                        add_triplet(triplets, idx_LH, idx_from, prob_z * wm_low * wa_high);
                        add_triplet(triplets, idx_HH, idx_from, prob_z * wm_high * wa_high);
                    }
                }
            }
        }

        SparseMat Lambda(grid.total_size, grid.total_size);
        Lambda.setFromTriplets(triplets.begin(), triplets.end());
        return Lambda;
    }

private:
    void add_triplet(std::vector<Triplet>& list, int row, int col, double val) {
        if (val > 1e-15) list.push_back(Triplet(row, col, val));
    }

    // Reuse find_weights logic from Aggregator
    void find_weights(const UnifiedGrid& g, double val, 
                      int& i_low, int& i_high, 
                      double& w_low, double& w_high) const {
        // Clamp
        if (val <= g.nodes.front()) {
            i_low = 0; i_high = 0; w_low = 1.0; w_high = 0.0; return;
        }
        if (val >= g.nodes.back()) {
            i_low = g.size - 1; i_high = g.size - 1; w_low = 1.0; w_high = 0.0; return;
        }

        auto it = std::lower_bound(g.nodes.begin(), g.nodes.end(), val);
        i_high = std::distance(g.nodes.begin(), it);
        i_low = i_high - 1;

        double x_low = g.nodes[i_low];
        double x_high = g.nodes[i_high];
        
        // w_high * x_high + w_low * x_low = val
        w_high = (val - x_low) / (x_high - x_low);
        w_low = 1.0 - w_high;
    }
};

} // namespace Monad
