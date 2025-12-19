
from .nodes import BlockNode

class GraphCompiler:
    """
    Translates the Visual Node Graph into a MonadModel configuration.
    """
    def __init__(self, scene):
        self.scene = scene

    def compile(self):
        """
        Traverses nodes and returns a config dict.
        """
        config = {
            'parameters': {},
            'blocks': [],
            'n_assets': 2 # Default to HANK
        }
        
        # 1. Collect Nodes
        nodes = [item for item in self.scene.items() if isinstance(item, BlockNode)]
        
        for node in nodes:
            # 2. Extract Parameters
            # Prefix params with node type if needed, or flat?
            # Monad usually expects flat 'parameters' dict
            if node.params:
                config['parameters'].update(node.params)
                
            # 3. Determine Structure
            if node.node_type == "household":
                # Determine asset count via properties if existing
                pass
            
            # 4. Topology (Wires)
            # This is where we'd build the DAG.
            # For now, we assume standard blocks are present and params define behavior.
            # In Phase 7+, we'd actually use the wires to define variable flow.
            
        print(f"[Compiler] Compiled {len(nodes)} nodes into config.")
        return config
