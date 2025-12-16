#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include "../UnifiedGrid.hpp"
#include "../Params.hpp"

class JsonLoader {
public:
    using json = nlohmann::json;

    static void load_model(const std::string& filepath, UnifiedGrid& grid, MonadParams& params) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open IR file: " + filepath);
        }

        json data = json::parse(f);

        std::cout << "[Monad::IO] Loading model: " << data.value("model_name", "Unknown") << std::endl;

        // 1. Load Parameters
        if (data.contains("parameters")) {
            for (auto& [key, val] : data["parameters"].items()) {
                if (val.is_number()) {
                    params.scalars[key] = val.get<double>();
                }
            }
        }

        // 2. Load Grid Definition
        if (data.contains("agents") && data["agents"].size() > 0 && data["agents"][0].contains("grids")) {
            // Assume single agent "Household" for v1.1
            if (data["agents"][0]["grids"].contains("asset_a")) {
                auto grid_def = data["agents"][0]["grids"]["asset_a"];
                
                std::string type = grid_def.value("type", "Uniform");
                int size = grid_def.value("size", 100);
                double min_val = grid_def.value("min", 0.0);
                double max_val = grid_def.value("max", 10.0);

                if (type == "Log-spaced") {
                    std::cout << "[Monad::IO] Initializing Log-spaced grid (n=" << size << ")" << std::endl;
                    grid.resize(size);
                    double curvature = grid_def.value("curvature", 1.0); // Assume existing if log-spaced?
                    if (curvature == 1.0) curvature = 2.0; // Default logic
                    
                    // Simple log grid implementation matches GridGenerator roughly
                    // x_i = min + (max - min) * ( (i/(n-1))^curve )
                    for(int i=0; i<size; ++i) {
                        double ratio = (double)i / (size - 1);
                        grid.nodes[i] = min_val + (max_val - min_val) * std::pow(ratio, curvature);
                    }
                } else {
                    std::cout << "[Monad::IO] Initializing Uniform grid (n=" << size << ")" << std::endl;
                    grid.resize(size);
                    for(int i=0; i<size; ++i) {
                         grid.nodes[i] = min_val + (max_val - min_val) * ((double)i / (size - 1));
                    }
                }
            }
        }
    }
};
