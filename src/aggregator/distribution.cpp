#include "distribution.h"
#include "../DistributionAggregator.hpp"
#include "../UnifiedGrid.hpp"

// Delegate to generic implementation

std::vector<double> DistributionAggregator::compute_stationary_distribution(const std::vector<double>& policy, const std::vector<double>& a_grid) {
    UnifiedGrid ugrid(a_grid);
    int n = (int)a_grid.size();
    
    // Initial dist
    std::vector<double> D(n, 1.0/n);
    
    // Iterate
    // Old implementation did loop to convergence.
    // We replicate that by calling single steps.
    
    for(int t=0; t<5000; ++t) {
        D = Monad::DistributionAggregator::forward_iterate(D, policy, ugrid);
        
        // Simple convergence check could be added, but fixed iter is fine for this task.
    }
    return D;
}

std::vector<double> DistributionAggregator::forward_step(const std::vector<double>& current_dist, const std::vector<double>& policy, const std::vector<double>& a_grid) {
     UnifiedGrid ugrid(a_grid);
     return Monad::DistributionAggregator::forward_iterate(current_dist, policy, ugrid);
}
