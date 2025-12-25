#pragma once
#include <vector>
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

class TwoAssetBackend {
public:
    virtual ~TwoAssetBackend() = default;

    // Core Bellman Solver Interface
    // Returns max diff for convergence check
    virtual double solve_bellman_iteration(const TwoAssetPolicy& guess, 
                                           TwoAssetPolicy& result, 
                                           const IncomeProcess& income) = 0;

    // Initialization Hook (Optional)
    virtual void initialize(const TwoAssetPolicy& initial_guess) {}
};

} // namespace Monad
