import functools
import numpy as np
from .api import Model

# Global Registry
_CURRENT_MODEL_CONFIG = {}
_SHOCK_REGISTRY = {}

def model_config(**kwargs):
    """
    Decorator/Function to register model configuration.
    Usage:
    @model_config(alpha=0.4, chi=1.0)
    class MyModel: pass
    """
    def decorator(cls):
        # Extract class attrs that don't start with _
        # Or just store kwargs if used as function call
        config = {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}
        _CURRENT_MODEL_CONFIG.update(config)
        return cls
    
    # Support direct call without class
    if kwargs:
        _CURRENT_MODEL_CONFIG.update(kwargs)
        return None
    return decorator

def shock(name=None):
    """
    Decorator to register a shock function.
    Function should return a dictionary of {variable: path}.
    """
    def decorator(func):
        key = name or func.__name__
        _SHOCK_REGISTRY[key] = func
        return func
    return decorator

def run(shock_func_or_name, T=50, model_type="two_asset", **kwargs):
    """
    DSL Runner.
    Executes a shock experiment using current config.
    """
    # Resolve Shock
    if callable(shock_func_or_name):
        shock_data = shock_func_or_name(T=T, **kwargs)
    elif shock_func_or_name in _SHOCK_REGISTRY:
        shock_data = _SHOCK_REGISTRY[shock_func_or_name](T=T, **kwargs)
    else:
        raise ValueError(f"Unknown shock: {shock_func_or_name}")
        
    # Merge Configs
    params = _CURRENT_MODEL_CONFIG.copy()
    
    # Initialize Model Orchestrator
    mdl = Model(model_type=model_type, T=T, params=params)
    mdl.initialize()
    
    # Run
    print(f"--- DSL: Running {model_type} Experiment ---")
    results = mdl.run_experiment(shock_data)
    
    return results

# Syntactic Sugar Helpers
def AR1(rho, sigma, T=50):
    return sigma * (rho ** np.arange(T))
