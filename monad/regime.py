import numpy as np

class Regime:
    """
    Represents a specific economic regime (e.g., ZLB, Fiscal Limit).
    Defined by a condition and a binding rule.
    """
    def __init__(self, name):
        self.name = name
        self.condition_func = None
        self.binding_value = None
        self.is_active = False

    def when(self, condition_func):
        """
        Define condition for regime activation.
        condition_func(local_vars) -> boolean array
        """
        self.condition_func = condition_func
        return self

    def bind(self, value):
        """
        Define the value/rule when binding.
        e.g., set interest rate i = 0.
        """
        self.binding_value = value
        return self

class RegimeManager:
    """
    Manages multiple regimes and applies them during simulation.
    """
    def __init__(self):
        self.regimes = {}

    def add(self, regime):
        self.regimes[regime.name] = regime

    def evaluate(self, variable_path, variable_name):
        """
        Check regimes relevant to 'variable_name' and return 
        (constrained_path, valid_indices).
        """
        constrained_path = variable_path.copy()
        is_constrained = np.zeros(len(variable_path), dtype=bool)

        for name, r in self.regimes.items():
            # Simply passing the path to condition
            mask = r.condition_func(variable_path)
            
            if np.any(mask):
                if r.binding_value is not None:
                    # Apply scalar binding value
                    constrained_path[mask] = r.binding_value
                    is_constrained = is_constrained | mask
                    
        return constrained_path, is_constrained

# Predefined Regimes
def ZLB_Regime():
    """Standard Zero Lower Bound on Nominal Rate."""
    return Regime("ZLB") \
           .when(lambda i: i < 0.0) \
           .bind(0.0)
