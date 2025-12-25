#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "runners.hpp"
#include "io/json_loader.hpp" 
// We include json_loader only for quick config peek or we let runners re-load.
// Better: Peek at config using a lightweight json parser or standard loader.

int main(int argc, char* argv[]) {
    try {
        std::string config_path = (argc > 1) ? argv[1] : "config.json";
        
        if (!std::filesystem::exists(config_path)) {
            // If default config missing, fallback to hardcoded defaults (legacy behavior)
            // But which mode? Default to 3D HANK.
            std::cout << "[WARN] Config file not found: " << config_path << std::endl;
            std::cout << "[INFO] Defaulting to Standard 3D HANK." << std::endl;
            return run_two_asset(config_path); 
        }

        // Peek Config to decide mode
        // Note: Full JSON parsing just to check 1 field. unique_ptr overhead minimal.
        // We reuse Monad::JsonLoader helpers if available or manual parse.
        // Let's rely on Python passing a specific field? No, binary should be robust.
        
        // Simple heuristic: Load file content, search for "n_assets": 3 or "belief": true
        // This avoids linking heavy JSON logic in main if possible, but we likely link it anyway.
        
        std::ifstream f(config_path);
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        
        // Robust Check for "n_assets": 3 or "n_assets" : 3
        // We look for "n_assets" then check next non-whitespace char
        size_t pos = content.find("n_assets");
        bool n_assets_3 = false;
        if (pos != std::string::npos) {
            // Scan forward for ':' and then '3'
            size_t cursor = pos + 8; // len("n_assets")
            while(cursor < content.size() && (content[cursor] == ' ' || content[cursor] == '\"' || content[cursor] == ':')) {
                cursor++;
            }
            if (cursor < content.size() && content[cursor] == '3') {
                n_assets_3 = true;
            }
        }

        bool has_crypto = n_assets_3 || (content.find("crypto") != std::string::npos);
        bool has_belief = (content.find("belief") != std::string::npos);
        bool experiment_tt = (content.find("experiment_tt") != std::string::npos);

        if (experiment_tt) {
            std::cout << "[Dispatch] Mode: Experiment TT (Jacobian Dump)" << std::endl;
            return run_experiment_tt(config_path);
        }
        else if (has_crypto) {
             std::cout << "[Dispatch] Mode: 5D HANK (Crypto)" << std::endl;
             // return run_five_asset(config_path); // Not integrated yet
             std::cout << "5D Mode Not Integrated in Main Dispatcher yet. Running 3D." << std::endl;
             return run_two_asset(config_path);
        }
        else if (has_belief) {
             std::cout << "[Dispatch] Mode: 4D HANK (Belief)" << std::endl;
             // return run_four_asset(config_path); // Not integrated yet
             std::cout << "4D Mode Not Integrated in Main Dispatcher yet. Running 3D." << std::endl;
             return run_two_asset(config_path);
        }
        else {
            std::cout << "[Dispatch] Mode: Standard 3D HANK" << std::endl;
            return run_two_asset(config_path);
        }

    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return 1;
    }
}
