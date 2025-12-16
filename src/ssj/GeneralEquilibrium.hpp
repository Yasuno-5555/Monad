#pragma once
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <string>
#include "SsjSolver3D.hpp"

namespace Monad {

class GeneralEquilibrium {
    SsjSolver3D& solver;
    int T; // Horizon

public:
    GeneralEquilibrium(SsjSolver3D& s, int horizon) : solver(s), T(horizon) {}

    // Solve for General Equilibrium Impulse Response
    // Input: Shock path to r_m (monetary policy)
    // Output: Paths for Y, C, etc.
    std::map<std::string, Eigen::VectorXd> solve_monetary_shock(const Eigen::VectorXd& dr_m) {
        
        // 1. Get Partial Jacobians from Household Block
        // J_C_rm: Direct effect of r_m on C
        // J_C_Y:  Effect of Income (Y) on C (via w*N)
        auto J = solver.compute_block_jacobians(T);
        
        // Ensure keys exist
        if(J.find("C") == J.end()) throw std::runtime_error("Jacobian for C missing");
        if(J["C"].find("rm") == J["C"].end()) throw std::runtime_error("Partials for rm missing");
        if(J["C"].find("w") == J["C"].end()) throw std::runtime_error("Partials for w missing"); // mapped to Y
        
        Eigen::MatrixXd J_C_rm = J["C"]["rm"];
        Eigen::MatrixXd J_C_Y  = J["C"]["w"]; // Assumption: wage rate w moves 1-to-1 with Output Y (linear production)

        // 2. Define GE System
        // Goods Market Clearing: dY = dC
        // dC = J_C_rm * dr_m + J_C_Y * dY
        // => dY - J_C_Y * dY = J_C_rm * dr_m
        // => (I - J_C_Y) * dY = J_C_rm * dr_m
        // => dY = (I - J_C_Y)^-1 * (J_C_rm * dr_m)

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(T, T);
        Eigen::MatrixXd A = I - J_C_Y;
        
        Eigen::VectorXd dC_partial = J_C_rm * dr_m;
        Eigen::VectorXd dY = A.colPivHouseholderQr().solve(dC_partial); // Linear Solve

        // 3. Recover other variables
        // dC = dY (Market Clearing)
        Eigen::VectorXd dC = dY;
        
        // decomposition: dC = Direct + Indirect(GE)
        // Direct = J_C_rm * dr_m
        // Indirect = J_C_Y * dY
        
        // Pack results
        std::map<std::string, Eigen::VectorXd> results;
        results["dr_m"] = dr_m;
        results["dY"] = dY;
        results["dC"] = dC;
        results["dC_direct"] = dC_partial;
        results["dC_indirect"] = J_C_Y * dY;
        
        return results;
    }
};

} // namespace Monad
