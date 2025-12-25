#pragma once
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <string>
#include "ssj/SsjSolver3D.hpp"
#include "blocks/FiscalBlock.hpp"

namespace Monad {

class FiscalExperiment {
    SsjSolver3D& solver;
    int T;
    
public:
    FiscalExperiment(SsjSolver3D& s, int horizon) : solver(s), T(horizon) {}
    
    // 1. Solve Fiscal Shock (Gov Spending G or Transfer T)
    // Assumes Monetary Policy keeps r constant (Partial Equilibrium) or Taylor Rule?
    // Let's implement General Equilibrium with Taylor Rule.
    struct FiscalResult {
        Eigen::VectorXd dY;
        Eigen::VectorXd dC;
        Eigen::VectorXd dG;
        Eigen::VectorXd dTrans;
        Eigen::VectorXd dr; // Interest rate response
        Eigen::VectorXd multiplier; // dY / dG
    };
    
    FiscalResult solve_fiscal_shock(const Eigen::VectorXd& dG_path, const Eigen::VectorXd& dTrans_path) {
        // 1. Get Jacobians
        auto J = solver.compute_block_jacobians(T);
        
        // J_C_r, J_C_w (Income), J_C_T (Transfer)
        // Note: Standard SsjSolver might not expose J_C_T by default if it wasn't requested.
        // But SsjSolver3D::compute_block_jacobians iterates ALL inputs in policy partials.
        // If "transfer" is in policy inputs, it will be in J.
        // We need to ensure "transfer" is treated as an input in JacobianBuilder.
        // This is implicit if the Policy function depends on Transfer.
        // In TwoAssetSolver, Transfer is part of tax rule parameters, usually fixed.
        // To make it time-varying, we need dC/dTrans.
        // Assuming J["C"]["trans"] exists or we approximate J_C_T ~ MPC * Distribution?
        // Let's assume for now we only have J_C_Y (Income) and we map Transfer -> Income directly.
        // Transfers T increase Net Income Z. dZ = dY - dT_tax + dTrans.
        // Net Income Z = Y - Tax(Y) + Trans.
        // dZ = (1 - Tax') * dY + dTrans.
        
        Eigen::MatrixXd J_C_r = J["C"]["rm"];
        Eigen::MatrixXd J_C_Z = J["C"]["w"]; // We usually map "w" to Net Income Z sensitivity in SSJ
        
        // Block: Net Income
        // dZ = (1 - tau) * dY + dTrans (Simplified linear tax)
        // Actually, let's just use dY for everything for simplicity in this experiment unless we have explicit tax block.
        // Let's assume dZ = dY + dTrans. (Lump sum transfer)
        
        // System:
        // Y = C + G
        // C = J_C_Z * dZ + J_C_r * dr
        // dZ = dY + dTrans
        // dr = phi_pi * pi ... (Taylor Rule)
        
        // Let's do simple Real Rate Rule for clarity: dr = 0 (Accommodation)
        // or Taylor Rule. Let's start with dr = 0.
        
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(T, T);
        
        // C = J_C_Z * (dY + dTrans)
        // Y = J_C_Z * (dY + dTrans) + dG
        // (I - J_C_Z) * dY = J_C_Z * dTrans + dG
        
        Eigen::VectorXd RHS = J_C_Z * dTrans_path + dG_path;
        Eigen::MatrixXd LHS = I - J_C_Z;
        
        Eigen::VectorXd dY = LHS.colPivHouseholderQr().solve(RHS);
        Eigen::VectorXd dC = J_C_Z * (dY + dTrans_path);
        
        // Multiplier
        Eigen::VectorXd mult = Eigen::VectorXd::Zero(T);
        for(int t=0; t<T; ++t) {
            double denom = std::abs(dG_path(t)) + std::abs(dTrans_path(t));
            if(denom > 1e-6) mult(t) = dY(t) / denom;
        }
        
        return {dY, dC, dG_path, dTrans_path, Eigen::VectorXd::Zero(T), mult};
    }
    
    // 2. Multiplier Decomposition
    // Decompose dC into Direct (PE) and Indirect (GE - Income)
    struct Decomposition {
        Eigen::VectorXd direct;    // Substitution + Direct Transfer effect
        Eigen::VectorXd indirect;  // Income effect (MPC * dY)
    };
    
    Decomposition decompose_multiplier(const Eigen::VectorXd& dY, const Eigen::VectorXd& dTrans, const Eigen::VectorXd& dr) {
        auto J = solver.compute_block_jacobians(T);
        Eigen::MatrixXd J_C_r = J["C"]["rm"];
        Eigen::MatrixXd J_C_Z = J["C"]["w"];
        
        // Net Income dZ = dY + dTrans
        // C = J_C_r * dr + J_C_Z * (dY + dTrans)
        
        // Direct Effect (Partial Eq):
        // Response to r and Transfer, holding Y constant (no aggregate income loop)
        Eigen::VectorXd direct = J_C_r * dr + J_C_Z * dTrans;
        
        // Indirect Effect (General Eq):
        // Response to endogenous Income change Y
        Eigen::VectorXd indirect = J_C_Z * dY;
        
        return {direct, indirect};
    }
};

} // namespace Monad
