#define NOMINMAX
#include "AnalyticalSolver.hpp"
#include "grid/grid_generator.h" 
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::ofstream out("solver_output.txt");
    std::streambuf *coutbuf = std::cout.rdbuf(); // Save old buf
    std::cout.rdbuf(out.rdbuf()); // Redirect cout to file

    try {
        std::cout << "Starting Solver Test..." << std::endl;
        
        // 1. Generate Grid
        std::vector<double> nodes = GridGenerator::generate("log_spaced", 0.0, 100.0, 100, 1.5);
        UnifiedGrid grid(nodes);
        
        // 2. Parameters
        double r_guess = 0.03; // Start higher? r < rho (rho ~ 2%)
        // With beta = 0.96 (standard yearly) => rho ~ 4%
        // Let's use beta = 0.96 as in user prompt example, r_guess 0.03
        double beta = 0.96;
        double sigma = 2.0; 
        
        // 3. Solve
        AnalyticalSolver::solve_steady_state(grid, r_guess, beta, sigma);
        
        std::cout << "Final Interest Rate: " << r_guess * 100.0 << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    std::cout.rdbuf(coutbuf); // Reset
    return 0;
}
