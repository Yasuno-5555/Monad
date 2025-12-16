#pragma once
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

struct IncomeProcess {
    int n_z;
    std::vector<double> z_grid;
    std::vector<double> Pi_flat; // Flattened transition matrix (row-major)

    // Helper to get Pi(i, j)
    double prob(int i, int j) const {
        return Pi_flat[i * n_z + j];
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
