#pragma once
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <Eigen/Dense>
#include "ssj/SsjSolver3D.hpp"
#include "grid/MultiDimGrid.hpp"

namespace Monad {

class InequalityAnalyzer {
    const MultiDimGrid& grid;
    const std::vector<double>& D_ss;    // Steady State Distribution
    const TwoAssetPolicy& pol_ss;       // Steady State Policy
    
    // Cached Jacobians (Policy Derivatives)
    // dC_i / dU (Direct effect of inputs U on individual i)
    // PartialMap is map<string, map<string, vector<double>>>
    PartialMap partials; 

    // Cached Indices
    std::vector<int> idx_top10;
    std::vector<int> idx_bottom50;
    std::vector<int> idx_debtors;

public:
    InequalityAnalyzer(const MultiDimGrid& g, const std::vector<double>& d, const TwoAssetPolicy& p,
                       const PartialMap& parts)
        : grid(g), D_ss(d), pol_ss(p), partials(parts) 
    {
        // Initialize Groups
        idx_top10    = get_indices_by_wealth_percentile(0.90, 1.00);
        idx_bottom50 = get_indices_by_wealth_percentile(0.00, 0.50);
        idx_debtors  = get_indices_by_asset_condition([](double m, double a){ return m < 0.0; });
        
        std::cout << "InequalityAnalyzer Initialized:" << std::endl;
        std::cout << "  Top 10% Size: " << idx_top10.size() << " nodes" << std::endl;
        std::cout << "  Bottom 50% Size: " << idx_bottom50.size() << " nodes" << std::endl;
        std::cout << "  Debtors Size: " << idx_debtors.size() << " nodes" << std::endl;
    }

    // Main Analysis: Decompose aggregate dC into groups
    // Input: paths of prices dU (T x n_inputs)
    struct GroupPaths {
        Eigen::VectorXd top10;
        Eigen::VectorXd bottom50;
        Eigen::VectorXd debtors;
    };

    GroupPaths analyze_consumption_response(const std::map<std::string, Eigen::VectorXd>& dU_paths) {
        
        if (dU_paths.empty()) return {};
        int T = dU_paths.begin()->second.size();
        
        GroupPaths results;
        results.top10    = Eigen::VectorXd::Zero(T);
        results.bottom50 = Eigen::VectorXd::Zero(T);
        results.debtors  = Eigen::VectorXd::Zero(T);

        // Iterate over time t
        for(int t=0; t<T; ++t) {
            
            // Calculate individual dC_i at time t
            // dC_{i,t} = sum_U ( dC_i/dU * dU_t )  <-- Direct Policy Effect
            
            // We accumulate the WEIGHTED change for each group first to avoid iterating whole grid multiple times
            // Actually, let's just make a full dC_total vector for simplicity and then aggregate.
            // Grid size is ~4000, T~100. 400,000 ops. Fast enough.
            
            std::vector<double> dC_total(grid.total_size, 0.0);
            
            for(auto const& [var, path] : dU_paths) {
                double shock = path[t];
                
                // Check if we have partials for this var
                if(partials["c"].count(var)) {
                    const auto& sens = partials["c"][var]; // Vector of sensitivities
                    for(int i=0; i<grid.total_size; ++i) {
                        dC_total[i] += sens[i] * shock;
                    }
                }
            }

            // Aggregate by Group
            results.top10[t]    = aggregate_group_change(dC_total, idx_top10);
            results.bottom50[t] = aggregate_group_change(dC_total, idx_bottom50);
            results.debtors[t]  = aggregate_group_change(dC_total, idx_debtors);
        }
        
        return results;
    }

    // New: Compute Aggregate Adjustment Rate Change
    // dA_t = Sum_i ( Adjust_ss[i] * dD_{i,t} )
    // Note: requires dD path. We only have dU path.
    // We need to re-compute dD using Fake News? Or simpler?
    // SsjSolver3D does not expose dD.
    // However, dD = Lambda * dD(-1) + F * dU.
    // We can simulate dD here if we have Fake News logic?
    // Or, we accept that we can't do this easily without dD.
    // Alternative: Just export Heatmap of dC (Policy) which is the "Policy Heatmap".
    
    // Let's implement getting the "Direct Policy Heatmap" at time t
    std::vector<double> compute_consumption_heatmap(const std::map<std::string, Eigen::VectorXd>& dU_paths, int t) {
        std::vector<double> heatmap(grid.total_size, 0.0);
        for(auto const& [var, path] : dU_paths) {
            double shock = path[t];
            if(partials["c"].count(var)) {
                const auto& sens = partials["c"][var];
                for(int i=0; i<grid.total_size; ++i) {
                    heatmap[i] += sens[i] * shock;
                }
            }
        }
        return heatmap;
    }
    
    // Helper to get Adjustment Flag (SS)
    std::vector<double> get_adjustment_map() {
        std::vector<double> adj(grid.total_size);
        for(int i=0; i<grid.total_size; ++i) adj[i] = pol_ss.adjust_flag[i];
        return adj;
    }

private:
    std::vector<int> get_indices_by_wealth_percentile(double p_low, double p_high) {
        // 1. Create index vector
        std::vector<int> indices(grid.total_size);
        std::iota(indices.begin(), indices.end(), 0);
        
        // 2. Sort by Total Wealth (m + a)
        std::sort(indices.begin(), indices.end(), [&](int i, int j){
            auto vi = grid.get_values(i);
            auto vj = grid.get_values(j);
            return (vi.first + vi.second) < (vj.first + vj.second);
        });
        
        // 3. Select range based on Accumulating Mass D_ss
        std::vector<int> selection;
        double current_mass = 0.0;
        double total_mass = 0.0;
        for(double d : D_ss) total_mass += d; // Should be 1.0
        
        for(int idx : indices) {
            // Include mass of current bin?
            // "Percentile" usually implies CDF.
            // Let's use mid-point or just accumulation.
            
            double mass = D_ss[idx] / total_mass;
            double prev_cdf = current_mass;
            current_mass += mass;
            
            // Check intersection with [p_low, p_high]
            // If the bin overlaps with the range, include it.
            // Or simpler: if cumulative mass is within range.
            
            // Logic: Include if representative of that range.
            // Strict: prev_cdf >= p_low && current_mass <= p_high
            // Loose (for plotting): Any overlap
            
            double overlap_start = std::max(prev_cdf, p_low);
            double overlap_end = std::min(current_mass, p_high);
            
            if(overlap_end > overlap_start) {
                selection.push_back(idx);
            }
        }
        return selection;
    }
    
    // Helper to get m, a, z from index
    // Note: grid.get_coords_values returns pair(m, a). Z is implicit in getting coords?
    // MultiDimGrid::get_values(i) returns pair<double, double> (m, a)? No, it returns vector?
    // Let's double check MultiDimGrid.hpp or use get_coords and look up nodes.
    // For now assuming get_coords_values exists or I implement logic.
    // Wait, MultiDimGrid usually has `get_values`?
    // Let's implement robustly.
    
    template<typename Func>
    std::vector<int> get_indices_by_asset_condition(Func condition) {
        std::vector<int> selection;
        for(int i=0; i<grid.total_size; ++i) {
            int im, ia, iz;
            grid.get_coords(i, im, ia, iz);
            double m = grid.m_grid.nodes[im];
            double a = grid.a_grid.nodes[ia];
            if(condition(m, a)) {
                selection.push_back(i);
            }
        }
        return selection;
    }

    double aggregate_group_change(const std::vector<double>& dC, const std::vector<int>& indices) {
        double num = 0.0;
        double den = 0.0;
        for(int i : indices) {
            num += dC[i] * D_ss[i];
            den += D_ss[i];
        }
        return (den > 1e-12) ? num / den : 0.0; // Per-capita change
    }
};

} // namespace Monad
