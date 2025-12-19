from .core import PolicyBlock

class InertialTaylor(PolicyBlock):
    def __init__(self, phi_pi=1.5, rho=0.8, name="central_bank"):
        super().__init__(name)
        # Default Parameters (stored in overrides initially or separate param dict)
        self.overrides['phi_pi'] = phi_pi
        self.overrides['rho'] = rho
        
        # Standard Equation Definition
        # Note: This string format is for documentation/future Parser interpretation.
        # In the Python Solver (PiecewiseSolver), we currently use hardcoded logic or lookups.
        # To make this effectively work in Phase 2 without a full equation parser,
        # we will allow the Solver to query `policy.get_rule_function()` or similar.
        
        self.equations['rule'] = f"i = rho * i(-1) + (1-rho) * (phi_pi * pi)"

    def evaluate_flow(self, context):
        # 1. Parsing Context
        pi = context.get('pi', 0.0)
        i_lag = context.get('i_lag', 0.0) # Solver must provide lag
        # r is needed for learning "r - r_belief"
        r_real = context.get('r', 0.0) 
        
        # 2. Parameters
        rho = self.overrides.get('rho', 0.0)
        phi_pi = self.overrides.get('phi_pi', 1.5)
        
        # 3. Learning Logic (Superset Strategy)
        updates = {}
        r_star_val = 0.0 # Default natural rate if not learned
        
        # Look for "r_star_belief" in learning regimes
        # self._learning_regimes keys are full names e.g. "Fed_r_star_belief"
        # We need to find if we are learning r_star.
        # Assuming single learned var for Phase 2 demo simplicity or loop.
        
        for full_var_name, strategies in self._learning_regimes.items():
            # Assume target is "r_star" if belief is "r_star_belief"
            # But we generated name as "{self.name}_r_star_belief"
            
            # Simple check: is this active?
            # User sets 'gain' in overrides. 
            # If gain > 0, we use Active eq. Logic is hardcoded here for demo efficiency.
            
            # Get lagged belief from context. 
            # Context key should match full_var_name
            belief_lag = context.get(f"{full_var_name}_lag", 0.0)
            
            gain = self.overrides.get('gain', 0.0) # Global gain for now
            
            if gain > 0.0:
                # Active Learning: b = b(-1) + gain * (r - b(-1))
                # Note: target "r" is r_real here.
                # In generic, we need to know what 'target_var' was.
                # For `learns("r_star")`, target is effectively r_real (tracking nature).
                belief_new = belief_lag + gain * (r_real - belief_lag)
            else:
                # Dormant: b = b(-1)
                belief_new = belief_lag
                
            updates[full_var_name] = belief_new
            r_star_val = belief_new # Use this in rule

        # 4. Taylor Rule with Belief
        # i = r* + rho*i(-1) + (1-rho)*(phi*pi)
        
        i_target = r_star_val + rho * i_lag + (1 - rho) * (phi_pi * pi)
        updates['i'] = i_target
        
        return updates

class VolckerRule(PolicyBlock):
    def __init__(self, phi_pi=5.0, name="central_bank"):
        super().__init__(name)
        self.overrides['phi_pi'] = phi_pi
        self.overrides['rho'] = 0.0
        self.equations['rule'] = "i = phi_pi * pi"

# Registry for Rules
class Rules:
    InertialTaylor = InertialTaylor
    Volcker = VolckerRule
