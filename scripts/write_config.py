
import json

config = {
    "rho_z": 0.95,
    "sigma_z": 0.2,
    "n_z": 7,
    "r_m": 0.01,
    "r_a": 0.02,
    "beta": 0.98,
    "sigma": 1.0,
    "m_min": 0.0,
    "chi": 0.05
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)
    
print("Wrote config.json")
