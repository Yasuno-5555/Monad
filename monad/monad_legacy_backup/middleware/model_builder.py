import sympy as sp
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ModelBuilder:
    name: str
    grids: Dict[str, dict] = field(default_factory=dict)
    variables: Dict[str, dict] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)
    egm_equations: List[dict] = field(default_factory=list)
    distribution_equations: List[dict] = field(default_factory=list)
    market_clearing_equations: List[dict] = field(default_factory=list)

    def add_grid(self, name: str, kind: str, min_val: float, max_val: float, size: int, curvature: float = 1.0):
        self.grids[name] = {
            "name": name, "kind": kind, "min_val": min_val, "max_val": max_val, 
            "size": size, "curvature": curvature
        }

    def add_variable(self, name: str, vtype: str, desc: str = "", bounds: tuple = (None, None)):
        self.variables[name] = {
            "name": name, "type": vtype, "description": desc,
            "min_val": bounds[0], "max_val": bounds[1]
        }
        # Return sympy symbol for convenience
        return sp.Symbol(name)

    def add_parameter(self, name: str, value: float):
        self.parameters[name] = value
        return sp.Symbol(name)

    def add_egm_equation(self, lhs: sp.Symbol, rhs: sp.Expr, condition: Optional[str] = None):
        self.egm_equations.append({
            "lhs": str(lhs),
            "rhs": str(rhs), # simplistic string conversion
            "condition": condition
        })
        
    def add_distribution_equation(self, lhs: sp.Symbol, rhs: sp.Expr):
        self.distribution_equations.append({
            "lhs": str(lhs),
            "rhs": str(rhs)
        })

    def add_market_clearing(self, lhs: sp.Symbol, rhs: sp.Expr):
         self.market_clearing_equations.append({
            "lhs": str(lhs),
            "rhs": str(rhs)
        })
