#pragma once
#include <memory>
#include <iostream>
#include <stdexcept>
#include "../backend/TwoAssetBackend.hpp"
#include "../grid/MultiDimGrid.hpp"
#include "../Params.hpp"

namespace Monad {

class ThreeAssetSolver {
    std::unique_ptr<ThreeAssetBackend> backend;

public:
    ThreeAssetSolver(std::unique_ptr<ThreeAssetBackend> backend_ptr) 
        : backend(std::move(backend_ptr)) {
        if (!backend) {
            throw std::runtime_error("ThreeAssetSolver: Backend cannot be null.");
        }
    }
    
    // Delegate to backend
    double solve_bellman(const ThreeAssetPolicy& guess, ThreeAssetPolicy& result, 
                         const IncomeProcess& income) {
        return backend->solve_bellman_iteration(guess, result, income);
    }
};

using TwoAssetSolver = ThreeAssetSolver;

} // namespace Monad
