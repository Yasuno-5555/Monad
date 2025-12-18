from typing import List, Dict, Union, Optional, Literal
from pydantic import BaseModel, Field

# --- Grid Definitions ---

class GridDef(BaseModel):
    name: str
    kind: Literal['uniform', 'log_spaced']
    min_val: float
    max_val: float
    size: int
    curvature: Optional[float] = 1.0 # For log grids, often parameterized

# --- Function Definitions ---

class Equation(BaseModel):
    lhs: str  # Left Hand Side (variable name)
    rhs: str  # Right Hand Side (SymPy expression as string)
    condition: Optional[str] = None # e.g. "a > 0" for non-negativity constraint

# --- Variable Definitions ---

class Variable(BaseModel):
    name: str
    type: Literal['state', 'control', 'parameter', 'shock']
    description: Optional[str] = ""
    # Optional bounds for validation
    min_val: Optional[float] = None
    max_val: Optional[float] = None

# --- Main Model Schema ---

class MonadIR(BaseModel):
    name: str
    version: str = "1.1"
    
    grids: List[GridDef]
    variables: List[Variable]
    
    # Separation of concerns
    parameters: Dict[str, float] # Default values
    
    # Model Logic
    # Equations are often grouped by stage in HANK models
    # e.g., Backward Step (EGM), Distribution Step, Aggregation
    
    egm_step: List[Equation] = Field(description="Equations for backward iteration (Euler)")
    distribution_step: List[Equation] = Field(description="Transition equations for distribution")
    market_clearing: List[Equation] = Field(description="Equations that must equal zero in equilibrium")

    class Config:
        title = "Monad Engine Intermediate Representation"
