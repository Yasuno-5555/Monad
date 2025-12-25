from .monad_core import Engine as _CppEngine
import numpy as np

class MonadEngine:
    """
    Stateful wrapper around the C++ Monad Engine.
    Enables fluent interface: model.param("beta", 0.98).solve()
    """
    def __init__(self, config=None):
        self._engine = _CppEngine()
        if config:
            self.setup(config)
            
    def setup(self, config):
        """Configure grid and income process."""
        if "grid" in config:
            g = config["grid"]
            self._engine.set_grid_config(
                g.get("Nm", 50), g.get("m_min", -2.0), g.get("m_max", 50.0), g.get("m_curv", 3.0),
                g.get("Na", 40), g.get("a_min", 0.0), g.get("a_max", 100.0), g.get("a_curv", 2.0)
            )
        
        if "income" in config:
            z = config["income"]["z_grid"]
            pi = config["income"]["Pi_flat"]
            self._engine.set_income_process(z, pi)
            
        if "params" in config:
            for k, v in config["params"].items():
                self._engine.set_param(k, float(v))
        return self

    def param(self, key, value):
        """Set a single parameter and return self for chaining."""
        self._engine.set_param(key, float(value))
        return self

    def solve(self, verbose=True):
        """Solve steady state using current parameters and state as guess."""
        if verbose:
            print(f"Solving SS... (beta={self._engine.get_param('beta'):.4f}, r_m={self._engine.get_param('r_m'):.4f})")
        
        self._engine.solve_steady_state()
        
        if verbose:
            agg_m = self._engine.compute_aggregate_liquid()
            agg_a = self._engine.compute_aggregate_illiquid()
            print(f"  Converged. Agg Liquid: {agg_m:.4f}, Agg Illiquid: {agg_a:.4f}")
            
        return self

    @property
    def result(self):
        """Export results as dictionary."""
        return {
            "policy_c": np.array(self._engine.get_policy_c()),
            "policy_m": np.array(self._engine.get_policy_m()),
            "policy_a": np.array(self._engine.get_policy_a()),
            "value": np.array(self._engine.get_value_function()),
            "distribution": np.array(self._engine.get_distribution()),
            "aggregates": {
                "liquid": self._engine.compute_aggregate_liquid(),
                "illiquid": self._engine.compute_aggregate_illiquid(),
                "consumption": self._engine.compute_aggregate_consumption()
            }
        }
    
    def plot_distribution(self):
        """Quick plot of the distribution (marginal)."""
        import matplotlib.pyplot as plt
        dist = np.array(self._engine.get_distribution())
        # Reshape? Need grid dims. For now just plot flat or use existing visualization tools.
        # This is a stub for the 10-line rule demo.
        plt.plot(dist)
        plt.title("Asset Distribution (Flat)")
        plt.show()
        return self
