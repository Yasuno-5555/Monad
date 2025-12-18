#pragma once
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

struct IncomeProcess {
    int n_z;
    std::vector<double> z_grid;
    std::vector<double> Pi_flat; // Flattened transition matrix (row-major)
    double unemployment_benefit = 0.0; // v1.7: Benefit level for z=0 state

    // Helper to get Pi(i, j)
    double prob(int i, int j) const {
        return Pi_flat[i * n_z + j];
    }
    
    // v1.7: Check if state is unemployment (z=0 is special)
    bool is_unemployed(int iz) const {
        // If unemployment_benefit > 0, then iz=0 is the unemployment state
        return (unemployment_benefit > 0.0 && iz == 0);
    }
    
    // v1.7: Get labor income for state iz
    // If unemployed: returns benefit (w-independent)
    // If employed: returns w * z[iz]
    double get_labor_income(int iz, double w) const {
        if (is_unemployed(iz)) {
            return unemployment_benefit * w; // Benefit as fraction of wage
        }
        return w * z_grid[iz];
    }
};

// Parameter container
struct MonadParams {
    std::map<std::string, double> scalars;
    IncomeProcess income;

    double get(const std::string& key, double default_val) const {
        auto it = scalars.find(key);
        if (it != scalars.end()) return it->second;
        return default_val;
    }
    
    double get_required(const std::string& key) const {
        auto it = scalars.find(key);
        if (it == scalars.end()) throw std::runtime_error("Missing required parameter: " + key);
        return it->second;
    }
};
