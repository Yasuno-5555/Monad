from dataclasses import dataclass, field
from typing import List, Dict, Union, Any
import yaml

@dataclass
class VariableSpec:
    name: str
    guess: float
    description: str = ""

@dataclass
class ParameterSpec:
    name: str
    value: float
    description: str = ""

@dataclass
class EquationSpec:
    resid: str
    description: str = ""

@dataclass
class BlockSpec:
    name: str
    type: str  # "heterogeneous", "simple", etc.
    kernel: str # e.g. "TwoAsset"
    inputs: List[str]
    outputs: List[str]

@dataclass
class ModelSpec:
    name: str
    type: str  # "static_equilibrium", "dsge_perfect_foresight"
    variables: Dict[str, VariableSpec]
    parameters: Dict[str, ParameterSpec]
    equations: List[EquationSpec]
    description: str = ""
    blocks: Dict[str, BlockSpec] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        vars_dict = {}
        for k, v in data.get('variables', {}).items():
            # Support simple format "Y: 100" or verbose "Y: {guess: 100, desc: ...}"
            if isinstance(v, (int, float)):
                vars_dict[k] = VariableSpec(k, float(v))
            else:
                vars_dict[k] = VariableSpec(k, float(v['guess']), v.get('description', ''))
                
        params_dict = {}
        for k, v in data.get('parameters', {}).items():
            if isinstance(v, (int, float)):
                params_dict[k] = ParameterSpec(k, float(v))
            else:
                params_dict[k] = ParameterSpec(k, float(v['value']), v.get('description', ''))
                
        equations = []
        for eq in data.get('equations', []):
            if isinstance(eq, str):
                equations.append(EquationSpec(eq))
            else:
                equations.append(EquationSpec(eq['resid'], eq.get('description', '')))
        
        blocks = {}
        for k, v in data.get('blocks', {}).items():
            blocks[k] = BlockSpec(
                name=k,
                type=v.get('type', 'heterogeneous'),
                kernel=v.get('kernel', ''),
                inputs=v.get('inputs', []),
                outputs=v.get('outputs', [])
            )
            
        return cls(
            name=data.get('name', 'Untitled'),
            type=data.get('type', 'static_equilibrium'),
            variables=vars_dict,
            parameters=params_dict,
            equations=equations,
            description=data.get('description', ''),
            blocks=blocks
        )

    @classmethod
    def from_yaml(cls, yaml_str: str):
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
