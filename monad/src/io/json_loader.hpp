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
            auto grid_def = data["agents"][0]["grids"]["asset_a"];
            
            std::string type = grid_def.value("type", "Uniform");
            int size = grid_def.value("n_points", 100); // Updated to match Python "n_points"
            if (size == 100 && grid_def.contains("size")) size = grid_def["size"]; // Backwards compat

            double min_val = grid_def.value("min", 0.0);
            double max_val = grid_def.value("max", 10.0);

            if (type == "Log-spaced") {
                std::cout << "[Monad::IO] Initializing Log-spaced grid (n=" << size << ")" << std::endl;
                grid.resize(size);
                double curv = grid_def.value("potency", 2.0); // Updated to match Python "potency"
                if (curv == 2.0 && grid_def.contains("curvature")) curv = grid_def["curvature"];

                for(int i=0; i<size; ++i) {
                    double ratio = (double)i / (size - 1);
                    grid.nodes[i] = min_val + (max_val - min_val) * std::pow(ratio, curv);
                }
            } else {
                std::cout << "[Monad::IO] Initializing Uniform grid (n=" << size << ")" << std::endl;
                grid.resize(size);
                for(int i=0; i<size; ++i) {
                        grid.nodes[i] = min_val + (max_val - min_val) * ((double)i / (size - 1));
                }
            }
        }

        // 3. Load Income Process (v1.2)
        if (data.contains("income_process")) {
            auto inc_def = data["income_process"];
            params.income.n_z = inc_def.value("n_z", 1);
            
            if (inc_def.contains("z_grid")) {
                 params.income.z_grid = inc_def["z_grid"].get<std::vector<double>>();
            } else {
                 params.income.z_grid = {1.0};
            }

            if (inc_def.contains("transition_matrix")) {
                 params.income.Pi_flat = inc_def["transition_matrix"].get<std::vector<double>>();
            } else {
                 params.income.Pi_flat = {1.0};
            }
            std::cout << "[Monad::IO] Loaded Income Process (Nz=" << params.income.n_z << ")" << std::endl;
            
            // v1.7: Load unemployment benefit if present
            if (data.contains("parameters") && data["parameters"].contains("unemployment_benefit")) {
                params.income.unemployment_benefit = data["parameters"]["unemployment_benefit"].get<double>();
                std::cout << "[Monad::IO] Unemployment Benefit: " << params.income.unemployment_benefit << std::endl;
            }
        } else {
            // Default to deterministic
            params.income.n_z = 1;
            params.income.z_grid = {1.0};
            params.income.Pi_flat = {1.0};
        }
    }
};
