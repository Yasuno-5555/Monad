#pragma once
#include <memory>
#include <iostream>
#include <stdexcept>
#include "../backend/TwoAssetBackend.hpp"
#include "../grid/MultiDimGrid.hpp"
#include "../Params.hpp"

namespace Monad {

class TwoAssetSolver {
    std::unique_ptr<TwoAssetBackend> backend;

public:
    TwoAssetSolver(std::unique_ptr<TwoAssetBackend> backend_ptr) 
        : backend(std::move(backend_ptr)) {
        if (!backend) {
            throw std::runtime_error("TwoAssetSolver: Backend cannot be null.");
        }
    }
    
    // Delegate to backend
    double solve_bellman(const TwoAssetPolicy& guess, TwoAssetPolicy& result, 
                         const IncomeProcess& income) {
        return backend->solve_bellman_iteration(guess, result, income);
    }
};

} // namespace Monad
