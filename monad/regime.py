from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict
import numpy as np

@dataclass
class RegimeDefinition:
    name: str
    condition: Callable  # The boolean mask generator func(model, guess_path) -> mask
    rule: Optional[Callable] = None # The structural override func(model) -> overrides dict
    priority: int = 0

class RegimeRegistry:
    def __init__(self):
        self._regimes: Dict[str, RegimeDefinition] = {}
        # Always register "normal" regime implicitly with lowest priority
        # Actually normal is fallback, might not need explicit registry entry if handled by solver default
    
    def register(self, name, condition, priority=0):
        reg = RegimeDefinition(name=name, condition=condition, priority=priority)
        self._regimes[name] = reg
        return reg

    def get_by_priority(self):
        # Sort by priority (Higher number = Higher priority)
        return sorted(self._regimes.values(), key=lambda x: x.priority, reverse=True)

    def evaluate_regimes(self, model, guess_path):
        """
        Evaluate all regimes and return the active regime name for each time step.
        Returns: array of strings (T,)
        """
        # Assume guess_path is dict of vectors? Or model knows how to extract?
        # Let's pass the raw guess vector/dict to the condition function.
        
        # 1. Start with "normal"
        # Determine T from guess. 
        # Guess might be a single vector or dict.
        # Let's assume generic T from model context if possible, or infer.
        if isinstance(guess_path, dict):
            T = len(next(iter(guess_path.values())))
        else: # Vector
            T = len(guess_path)
            
        active_regimes = np.array(["normal"] * T, dtype=object)
        
        # 2. Iterate by priority (ascending or descending?)
        # We want Highest Priority to OVERWRITE lower ones.
        # So Iterate Low -> High? Or High -> Low and only write if "normal"?
        # User said: "Crisis > ZLB > Normal".
        # If we iterate High Priority first, we write on "normal". Then subsequent lower priorities shouldn't overwrite?
        # NO.
        # If ZLB is prio 5, Crisis is prio 10.
        # Crisis overwrites normal. ZLB overwrites normal?
        # If both are true... Crisis should win.
        # So iterate Low -> High?
        # ZLB (5) writes. Crisis (10) overwrites ZLB. YES.
        
        # Sort Low to High
        sorted_regimes = sorted(self._regimes.values(), key=lambda x: x.priority)
        
        for r in sorted_regimes:
            try:
                # condition(model, guess_path) -> bool array
                mask = r.condition(model, guess_path)
                if np.any(mask):
                    active_regimes[mask] = r.name
            except Exception as e:
                print(f"[Warn] Regime {r.name} evaluation failed: {e}")
                
        return active_regimes

# Global singleton
registry = RegimeRegistry()

def regime(priority=1):
    """Decorator to define a new regime."""
    def decorator(func):
        # Register the condition function
        reg_def = registry.register(func.__name__, func, priority)
        
        # Attach the .rule decorator to the function itself
        def rule_decorator(rule_func):
            reg_def.rule = rule_func
            return rule_func
        
        func.rule = rule_decorator
        return func
    return decorator

# Re-export legacy/manual classes if needed, or deprecate previous 'Regime' class
# For compatibility with Advanced Features Test, let's keep ZLB_Regime wrapper?
# Or update the test later. The user wants the NEW API.
# But MonadModel might expect the old 'RegimeManager'.
# Let's Alias or Adapt if necessary.
# The previous `RegimeManager` in `monad/regime.py` was simpler. 
# We are replacing `monad/regime.py` content. We should ensure backward compat if possible
# or just declare a breaking change (Phase 1).
# I'll implement a Compatibility Layer for `ZLB_Regime` if I wiped it.
class LegacyRegimeAdapter:
    pass # Todo if needed
