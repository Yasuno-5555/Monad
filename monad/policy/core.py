from typing import Dict, Tuple, Optional, Any

class PolicyBlock:
    """
    Base class for Policy Agents.
    Encapsulates equations, parameters, and internal state variables.
    """
    def __init__(self, name="central_bank"):
        self.name = name
        self.auxiliary_vars: Dict[str, str] = {}
        self.equations: Dict[str, str] = {}
        self.overrides: Dict[str, Any] = {}
        self._model_ref = None # Link to parent model (Observer Pattern)
        self._learning_regimes: Dict[str, Dict[str, str]] = {} # target -> {active, dormant}

    def link_to_model(self, model):
        """Called by MonadModel.attach to establish link."""
        self._model_ref = model

    def declare_variable(self, name, latex=None):
        """Register a state variable required by this policy."""
        full_name = f"{self.name}_{name}"
        self.auxiliary_vars[full_name] = latex
        return full_name

    def learns(self, target_var, mechanism="constant_gain", gain=0.05):
        """
        Enable learning for a variable.
        Strategies:
        - Active: x_belief = x_belief(-1) + gain * (x - x_belief(-1))
        - Dormant: x_belief = x_belief(-1)
        """
        belief_var = f"{target_var}_belief"
        self.declare_variable(belief_var, latex=f"{target_var}^{{e}}")
        
        # Define equations strings
        # Full name usage might be needed depending on scoping, but here we assume local names based on declare_variable
        # Note: In a real system, we'd need to prefix `target_var` if it's not global, but assuming global for "r" or "pi".
        
        # Case A: Active Learning
        # Using simple string formatting. 
        # Note: {self.name}_{belief_var} is the internal name, but declare_variable returns the unique name.
        # Let's use the return value of declare_variable.
        # However, declare_variable was called above. Let's reconstruct or catch it?
        # declare_variable prepends self.name.
        
        full_belief_var = f"{self.name}_{belief_var}"
        
        eq_active = f"{full_belief_var} = {full_belief_var}(-1) + {gain} * ({target_var} - {full_belief_var}(-1))"
        eq_dormant = f"{full_belief_var} = {full_belief_var}(-1)"
        
        self._learning_regimes[full_belief_var] = {
            'active': eq_active,
            'dormant': eq_dormant
        }
        return self

    def export(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Start with standard equations
        exported_eqs = self.equations.copy()
        
        # Add Dormant equations for learning variables by default (Superset Strategy)
        for var_name, strategies in self._learning_regimes.items():
            exported_eqs[f"eq_{var_name}"] = strategies['dormant']
            
        return self.auxiliary_vars, exported_eqs
        
    def mutate(self, **kwargs):
        """Fluent configuration of parameters."""
        self.overrides.update(kwargs)
        # Propagate to model if linked
        if self._model_ref:
            self._model_ref.overrides.update(kwargs)
            # Also log change in model history if possible
            if hasattr(self._model_ref, '_log_change'):
                self._model_ref._log_change(self.name, kwargs)
        return self
    
    def get_overrides(self):
        return self.overrides

    def evaluate_flow(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Numerical evaluation of the policy logic.
        Args:
            context: Dict containing 'pi', 'i_lag', 'Y', etc. path values at time t.
        Returns:
            Dict containing 'i' (target) and any updated belief variables.
        """
        # Base implementation: return empty or pass
        return {}
