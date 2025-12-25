#pragma once
#include <vector>
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

class ThreeAssetBackend {
public:
    virtual ~ThreeAssetBackend() = default;

    // Core Bellman Solver Interface
    virtual double solve_bellman_iteration(const ThreeAssetPolicy& guess, 
                                           ThreeAssetPolicy& result, 
                                           const IncomeProcess& income) = 0;

    virtual void initialize(const ThreeAssetPolicy& initial_guess) {}
};

using TwoAssetBackend = ThreeAssetBackend;

} // namespace Monad
