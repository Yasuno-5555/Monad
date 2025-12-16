#pragma once
#include <map>
#include <string>
#include <stdexcept>

// Parameter container
struct MonadParams {
    std::map<std::string, double> scalars;
    
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
