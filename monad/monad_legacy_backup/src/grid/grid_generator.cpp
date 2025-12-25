#include "grid_generator.h"
#include <algorithm>
#include <iostream>

std::vector<double> GridGenerator::generate(const char* kind_cstr, double min_val, double max_val, int size, double curvature) {
    std::string kind(kind_cstr);
    if (size < 2) {
        throw std::invalid_argument("Grid size must be at least 2");
    }
    if (kind == "uniform") {
        return uniform(min_val, max_val, size);
    } else if (kind == "log_spaced") {
        return log_spaced(min_val, max_val, size, curvature);
    } else {
        throw std::invalid_argument("Unknown grid kind: " + kind);
    }
}

std::vector<double> GridGenerator::uniform(double min_val, double max_val, int size) {
    std::vector<double> grid(size);
    double step = (max_val - min_val) / (size - 1);
    for (int i = 0; i < size; ++i) {
        grid[i] = min_val + i * step;
    }
    return grid;
}

std::vector<double> GridGenerator::log_spaced(double min_val, double max_val, int size, double curvature) {
    std::vector<double> grid(size);
    double dist = max_val - min_val;
    double denom = std::exp(curvature) - 1.0;
    
    // x_i = min + dist * (exp(curv * i / (N-1)) - 1) / (exp(curv) - 1)
    
    for (int i = 0; i < size; ++i) {
        double ratio = static_cast<double>(i) / (size - 1);
        double num = std::exp(curvature * ratio) - 1.0;
        grid[i] = min_val + dist * (num / denom);
    }
    
    // Ensure bounds are exact
    grid[0] = min_val;
    grid[size - 1] = max_val;
    
    return grid;
}
