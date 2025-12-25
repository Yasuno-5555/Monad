import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
import os

class GPUBackend:
    """
    Handles data loading from C++ GPU Export.
    Converts raw Impulse Response vectors into Sequence Space Jacobian (Toeplitz) matrices.
    """
    def __init__(self, path_R, path_Z, T=50):
        self.T = T
        # Load and Create Toeplitz Matrices
        # J_C_r: Jacobian of Consumption w.r.t. Real Interest Rate
        self.J_C_r = self._load_jacobian(path_R, 'dC')
        # J_C_y: Jacobian of Consumption w.r.t. Income (MPC channel)
        # Note: Z shock implies an aggregate income shift Y.
        self.J_C_y = self._load_jacobian(path_Z, 'dC')
        
    def _load_jacobian(self, path, col_name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run C++ Engine first.")
            
        try:
            df = pd.read_csv(path)
            # Take first T elements of the impulse response
            # Assuming the CSV contains the single column IRF
            vec = df[col_name].values
            if len(vec) < self.T:
                # Pad with zeros if short (though unlikely for valid solver output)
                vec = np.pad(vec, (0, self.T - len(vec)))
            else:
                vec = vec[:self.T]
                
            # Create Lower Triangular Toeplitz Matrix
            # Column 0 is the impulse response.
            # Row 0 is [IRF[0], 0, 0, ...]
            # This maps input dX to output dY
            return toeplitz(vec, np.zeros(self.T))
        except Exception as e:
            raise RuntimeError(f"Failed to process {path}: {e}")
