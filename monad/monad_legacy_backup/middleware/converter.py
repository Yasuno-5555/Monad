from .ir_schema import MonadIR, GridDef, Variable, Equation
from .model_builder import ModelBuilder

def convert_to_ir(builder: ModelBuilder) -> MonadIR:
    grids = [
        GridDef(**g) for g in builder.grids.values()
    ]
    
    variables = [
        Variable(**v) for v in builder.variables.values()
    ]
    
    egm_step = [
        Equation(**eq) for eq in builder.egm_equations
    ]
    
    dist_step = [
        Equation(**eq) for eq in builder.distribution_equations
    ]
    
    mkt_step = [
        Equation(**eq) for eq in builder.market_clearing_equations
    ]
    
    return MonadIR(
        name=builder.name,
        grids=grids,
        variables=variables,
        parameters=builder.parameters,
        egm_step=egm_step,
        distribution_step=dist_step,
        market_clearing=mkt_step
    )
