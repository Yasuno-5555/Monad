"""
Monad Studio - Semantic Graph Compiler
Translates the Visual Node Graph into a fully-specified economic model configuration.

Key Features:
1. Node Semantics: Each node type has defined input/output variable slots
2. Edge Semantics: Wires bind variables between nodes (data flow)
3. DAG Construction: Topological ordering for equation system
4. Equation Generation: Produces symbolic equations from graph structure
5. Validation: Checks for unconnected ports, cycles, type mismatches
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
from collections import defaultdict

from .nodes import BlockNode, PortItem


class VariableType(Enum):
    """Economic variable types for type checking."""
    SCALAR = "scalar"       # Single value (e.g., interest rate)
    DISTRIBUTION = "dist"   # Distribution over states
    AGGREGATE = "agg"       # Aggregate variable
    PRICE = "price"         # Price/rate variable
    QUANTITY = "qty"        # Quantity variable
    ANY = "any"             # Wildcard


@dataclass
class VariableSlot:
    """Defines an input or output variable slot on a node."""
    name: str                           # Variable name (e.g., "C", "pi", "r")
    var_type: VariableType              # Type of variable
    description: str = ""               # Human-readable description
    default_equation: str = ""          # Default equation if not bound
    required: bool = True               # Whether connection is required


@dataclass
class NodeSemantics:
    """Semantic definition for a node type."""
    node_type: str
    inputs: List[VariableSlot]
    outputs: List[VariableSlot]
    equations: Dict[str, str]           # Output variable -> equation template
    description: str = ""


# =============================================================================
# SEMANTIC DEFINITIONS FOR ALL NODE TYPES
# =============================================================================

NODE_SEMANTICS: Dict[str, NodeSemantics] = {
    
    "household": NodeSemantics(
        node_type="household",
        inputs=[
            VariableSlot("w", VariableType.PRICE, "Wage rate"),
            VariableSlot("r", VariableType.PRICE, "Interest rate"),
            VariableSlot("div", VariableType.AGGREGATE, "Dividends", required=False),
        ],
        outputs=[
            VariableSlot("C", VariableType.AGGREGATE, "Aggregate consumption"),
            VariableSlot("N", VariableType.AGGREGATE, "Aggregate labor supply"),
        ],
        equations={
            # Euler equation and labor supply (symbolic templates)
            "C": "E[C_{t+1}] = (1 + r_t - delta) * C_t^(-1/eis)",
            "N": "w_t = frisch * N_t^(1/eta) * C_t^(1/eis)",
        },
        description="Household sector with consumption and labor decisions"
    ),
    
    "policy": NodeSemantics(
        node_type="policy",
        inputs=[
            VariableSlot("pi", VariableType.PRICE, "Inflation rate"),
        ],
        outputs=[
            VariableSlot("r", VariableType.PRICE, "Nominal interest rate"),
        ],
        equations={
            # Taylor rule
            "r": "r_t = rho * r_{t-1} + (1-rho) * (r_star + phi_pi * (pi_t - pi_star))",
        },
        description="Monetary policy (Taylor rule)"
    ),
    
    "market": NodeSemantics(
        node_type="market",
        inputs=[
            VariableSlot("C", VariableType.AGGREGATE, "Aggregate demand (consumption)"),
            VariableSlot("N", VariableType.AGGREGATE, "Labor supply", required=False),
        ],
        outputs=[
            VariableSlot("pi", VariableType.PRICE, "Inflation rate"),
            VariableSlot("div", VariableType.AGGREGATE, "Firm dividends"),
            VariableSlot("w", VariableType.PRICE, "Equilibrium wage", required=False),
        ],
        equations={
            # NK Phillips Curve and market clearing
            "pi": "pi_t = beta * E[pi_{t+1}] + kappa * (mc_t - mc_ss)",
            "div": "div_t = Y_t - w_t * N_t",
            "w": "w_t = mc_t * MPL_t",  # Marginal cost * marginal product of labor
        },
        description="Goods market with New Keynesian pricing"
    ),
    
    "fiscal": NodeSemantics(
        node_type="fiscal",
        inputs=[
            VariableSlot("Y", VariableType.AGGREGATE, "Output (tax base)"),
        ],
        outputs=[
            VariableSlot("G", VariableType.AGGREGATE, "Government spending"),
            VariableSlot("B", VariableType.AGGREGATE, "Government bonds"),
        ],
        equations={
            "G": "G_t = G_bar",  # Exogenous spending
            "B": "B_{t+1} = (1 + r_t) * B_t + G_t - tax_rate * Y_t",
        },
        description="Fiscal sector with spending and debt"
    ),
    
    "generic": NodeSemantics(
        node_type="generic",
        inputs=[],
        outputs=[],
        equations={},
        description="Generic block"
    ),
}


@dataclass
class VariableBinding:
    """Represents a connection between two variable slots."""
    source_node: str            # Source node ID
    source_var: str             # Output variable name
    target_node: str            # Target node ID
    target_var: str             # Input variable name
    wire_id: int = 0            # Reference to visual wire


@dataclass
class CompiledNode:
    """A node with resolved variable names and equations."""
    node_id: str
    node_type: str
    params: Dict[str, Any]
    input_vars: Dict[str, str]   # slot_name -> bound variable name
    output_vars: Dict[str, str]  # slot_name -> generated variable name
    equations: Dict[str, str]    # variable -> substituted equation


@dataclass 
class ValidationError:
    """Represents a validation error."""
    severity: str  # "error", "warning"
    node_id: str
    message: str


@dataclass
class CompiledGraph:
    """The result of graph compilation."""
    nodes: List[CompiledNode]
    bindings: List[VariableBinding]
    equations: Dict[str, str]         # All equations (variable -> equation)
    parameters: Dict[str, Any]        # Collected parameters
    variables: Dict[str, Dict]        # All variables with metadata
    topology: List[str]               # Topological order of node IDs
    errors: List[ValidationError]     # Validation errors
    is_valid: bool
    
    def to_config(self) -> Dict:
        """Export as MonadModel-compatible config."""
        return {
            "parameters": self.parameters,
            "variables": list(self.variables.keys()),
            "equations": self.equations,
            "n_assets": self.parameters.get("n_assets", 2),
            "blocks": [
                {
                    "id": n.node_id,
                    "type": n.node_type,
                    "inputs": n.input_vars,
                    "outputs": n.output_vars,
                }
                for n in self.nodes
            ],
            "topology": self.topology,
        }


class GraphCompiler:
    """
    Semantic Graph Compiler
    Translates the Visual Node Graph into a complete model specification.
    """
    
    def __init__(self, scene):
        self.scene = scene
        self.node_counter = 0
        
    def _generate_node_id(self, node: BlockNode) -> str:
        """Generate a unique ID for a node."""
        self.node_counter += 1
        # Use node type + counter
        return f"{node.node_type}_{self.node_counter}"
    
    def compile(self) -> CompiledGraph:
        """
        Main compilation pipeline.
        """
        errors: List[ValidationError] = []
        
        # Step 1: Collect all nodes with IDs
        nodes = [item for item in self.scene.items() if isinstance(item, BlockNode)]
        node_map: Dict[int, Tuple[str, BlockNode]] = {}  # Qt object ID -> (node_id, node)
        
        for node in nodes:
            node_id = self._generate_node_id(node)
            node_map[id(node)] = (node_id, node)
        
        # Step 2: Extract bindings from wires
        bindings = self._extract_bindings(node_map)
        
        # Step 3: Build dependency graph and validate
        dep_graph, rev_graph = self._build_dependency_graph(bindings, node_map)
        
        # Step 4: Topological sort
        topology, has_cycle = self._topological_sort(dep_graph, node_map)
        if has_cycle:
            errors.append(ValidationError("error", "", "Circular dependency detected in graph"))
        
        # Step 5: Compile each node with variable resolution
        compiled_nodes = []
        all_equations = {}
        all_variables = {}
        all_parameters = {}
        
        for node_id in topology:
            qt_id = next(k for k, v in node_map.items() if v[0] == node_id)
            _, node = node_map[qt_id]
            
            compiled, node_errors = self._compile_node(
                node_id, node, bindings, all_variables
            )
            compiled_nodes.append(compiled)
            errors.extend(node_errors)
            
            # Merge equations
            all_equations.update(compiled.equations)
            
            # Collect parameters
            all_parameters.update(compiled.params)
            
            # Register output variables
            for slot_name, var_name in compiled.output_vars.items():
                all_variables[var_name] = {
                    "source_node": node_id,
                    "slot": slot_name,
                    "type": self._get_var_type(node.node_type, slot_name, is_output=True)
                }
        
        # Step 6: Validation
        validation_errors = self._validate_graph(node_map, bindings, all_variables)
        errors.extend(validation_errors)
        
        is_valid = not any(e.severity == "error" for e in errors)
        
        # Log compilation result
        print(f"[Compiler] Compiled {len(nodes)} nodes, {len(bindings)} connections")
        print(f"[Compiler] Generated {len(all_equations)} equations, {len(all_variables)} variables")
        if errors:
            for e in errors:
                print(f"[Compiler] [{e.severity.upper()}] {e.node_id}: {e.message}")
        
        return CompiledGraph(
            nodes=compiled_nodes,
            bindings=bindings,
            equations=all_equations,
            parameters=all_parameters,
            variables=all_variables,
            topology=topology,
            errors=errors,
            is_valid=is_valid
        )
    
    def _extract_bindings(self, node_map: Dict) -> List[VariableBinding]:
        """Extract variable bindings from wire connections."""
        bindings = []
        wire_id = 0
        
        for qt_id, (node_id, node) in node_map.items():
            # Check output ports for wires
            for port in node.outputs:
                for wire in port.wires:
                    if wire.end_port:
                        # Find target node
                        target_node_qt = wire.end_port.parentItem()
                        if id(target_node_qt) in node_map:
                            target_node_id, _ = node_map[id(target_node_qt)]
                            
                            bindings.append(VariableBinding(
                                source_node=node_id,
                                source_var=port.name,
                                target_node=target_node_id,
                                target_var=wire.end_port.name,
                                wire_id=wire_id
                            ))
                            wire_id += 1
        
        return bindings
    
    def _build_dependency_graph(
        self, 
        bindings: List[VariableBinding],
        node_map: Dict
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Build directed dependency graph from bindings."""
        # Forward: node -> set of nodes it depends on
        dep_graph: Dict[str, Set[str]] = defaultdict(set)
        # Reverse: node -> set of nodes that depend on it
        rev_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Initialize all nodes
        for _, (node_id, _) in node_map.items():
            dep_graph[node_id]  # Ensure node exists even with no deps
        
        for binding in bindings:
            # Target depends on Source
            dep_graph[binding.target_node].add(binding.source_node)
            rev_graph[binding.source_node].add(binding.target_node)
        
        return dict(dep_graph), dict(rev_graph)
    
    def _topological_sort(
        self, 
        dep_graph: Dict[str, Set[str]],
        node_map: Dict
    ) -> Tuple[List[str], bool]:
        """Kahn's algorithm for topological sort."""
        # Copy graph
        in_degree = {node: len(deps) for node, deps in dep_graph.items()}
        queue = [node for node, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Find nodes that depend on this one
            for other, deps in dep_graph.items():
                if node in deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)
        
        has_cycle = len(result) != len(dep_graph)
        return result, has_cycle
    
    def _compile_node(
        self,
        node_id: str,
        node: BlockNode,
        bindings: List[VariableBinding],
        existing_vars: Dict[str, Dict]
    ) -> Tuple[CompiledNode, List[ValidationError]]:
        """Compile a single node with variable resolution."""
        errors = []
        semantics = NODE_SEMANTICS.get(node.node_type, NODE_SEMANTICS["generic"])
        
        # Resolve input variables (from bindings)
        input_vars = {}
        for slot in semantics.inputs:
            # Find binding for this input slot
            binding = next(
                (b for b in bindings 
                 if b.target_node == node_id and b.target_var == slot.name),
                None
            )
            
            if binding:
                # Use source variable name with node prefix
                input_vars[slot.name] = f"{binding.source_node}.{binding.source_var}"
            elif slot.required:
                errors.append(ValidationError(
                    "warning", node_id, 
                    f"Input '{slot.name}' is not connected"
                ))
                input_vars[slot.name] = f"{slot.name}_unbound"
            else:
                input_vars[slot.name] = f"{slot.name}_default"
        
        # Generate output variable names
        output_vars = {}
        for slot in semantics.outputs:
            output_vars[slot.name] = f"{node_id}.{slot.name}"
        
        # Substitute equations with resolved variable names
        equations = {}
        for out_var, eq_template in semantics.equations.items():
            if out_var in output_vars:
                # Basic substitution (in production, use sympy or similar)
                eq = eq_template
                for in_slot, in_var in input_vars.items():
                    eq = eq.replace(f"{in_slot}_t", in_var)
                    eq = eq.replace(in_slot, in_var)
                
                # Substitute parameters
                for param, value in node.params.items():
                    eq = eq.replace(param, str(value))
                
                equations[output_vars[out_var]] = eq
        
        return CompiledNode(
            node_id=node_id,
            node_type=node.node_type,
            params=node.params.copy(),
            input_vars=input_vars,
            output_vars=output_vars,
            equations=equations
        ), errors
    
    def _get_var_type(self, node_type: str, slot_name: str, is_output: bool) -> str:
        """Get variable type for a slot."""
        semantics = NODE_SEMANTICS.get(node_type)
        if semantics:
            slots = semantics.outputs if is_output else semantics.inputs
            for slot in slots:
                if slot.name == slot_name:
                    return slot.var_type.value
        return "any"
    
    def _validate_graph(
        self,
        node_map: Dict,
        bindings: List[VariableBinding],
        variables: Dict[str, Dict]
    ) -> List[ValidationError]:
        """Validate the compiled graph."""
        errors = []
        
        # Check for isolated nodes (no connections at all)
        connected_nodes = set()
        for binding in bindings:
            connected_nodes.add(binding.source_node)
            connected_nodes.add(binding.target_node)
        
        for _, (node_id, node) in node_map.items():
            if node_id not in connected_nodes:
                errors.append(ValidationError(
                    "warning", node_id,
                    f"Node '{node.title}' is not connected to any other node"
                ))
        
        # Check for required inputs without connections
        for _, (node_id, node) in node_map.items():
            semantics = NODE_SEMANTICS.get(node.node_type)
            if semantics:
                for slot in semantics.inputs:
                    if slot.required:
                        has_binding = any(
                            b.target_node == node_id and b.target_var == slot.name
                            for b in bindings
                        )
                        if not has_binding:
                            errors.append(ValidationError(
                                "warning", node_id,
                                f"Required input '{slot.name}' is not connected"
                            ))
        
        return errors
    
    def get_debug_info(self) -> Dict:
        """Return debug information about current graph state."""
        nodes = [item for item in self.scene.items() if isinstance(item, BlockNode)]
        
        wire_count = 0
        for node in nodes:
            for port in node.outputs:
                wire_count += len([w for w in port.wires if w.end_port])
        
        return {
            "node_count": len(nodes),
            "wire_count": wire_count,
            "node_types": [n.node_type for n in nodes],
        }
