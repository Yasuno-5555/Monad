#ifndef MONAD_DISTRIBUTION_AGGREGATOR_H
#define MONAD_DISTRIBUTION_AGGREGATOR_H

#include <vector>

class DistributionAggregator {
public:
    // Compute stationary distribution using Young's method (lottery)
    // Input: policy (next period asset choices on grid), a_grid
    // Output: probability mass on each grid point summing to 1
    static std::vector<double> compute_stationary_distribution(const std::vector<double>& policy, const std::vector<double>& a_grid);

private:
    static std::vector<double> forward_step(const std::vector<double>& current_dist, const std::vector<double>& policy, const std::vector<double>& a_grid);
};

#endif // MONAD_DISTRIBUTION_AGGREGATOR_H
