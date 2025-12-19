
import sys
from PySide6.QtWidgets import QApplication
from monad.gui.app import NodeEditor
from monad.gui.nodes import BlockNode

def test_gui_advanced():
    # 1. Instantiate
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    window = NodeEditor()
    print("[TEST] Window Instantiated.")
    
    # 2. Add Nodes (Simulate Buttons)
    print("[TEST] Adding Nodes...")
    window.scene.clear() # Prevent Demo nodes from interfering
    window.scene.nodes = [] # Clear python list tracking
    
    h = window.add_node("Household", ["r"], ["C"], (0,0), "household", {'beta': 0.99})
    cb = window.add_node("Central Bank", ["pi"], ["r"], (200,0), "policy", {'phi': 1.5})
    
    assert len(window.scene.nodes) == 2, f"Expected 2 nodes, got {len(window.scene.nodes)}"
    
    # 3. Test Property Editor (Simulate Selection)
    print("[TEST] Testing Property Editor...")
    
    # Select Household
    h.setSelected(True)
    window.on_selection_changed()
    
    # Check if Editor populated
    editor = window.prop_editor
    assert editor.current_node == h, "Editor did not pick up selection."
    
    # Simulate Edit: Change beta to 0.95
    spin_beta = editor.widgets.get('beta')
    assert spin_beta is not None, "Widget for 'beta' not created."
    
    spin_beta.setValue(0.95)
    # Check if Node param updated
    assert h.params['beta'] == 0.95, f"Param update failed. Expected 0.95, got {h.params['beta']}"
    print("[TEST] Property Edit Confirmed.")
    
    # 4. Test Wiring (Simulate Connection)
    print("[TEST] Testing Wiring & Compilation...")
    # Compiler Logic: Does it find the nodes and params?
    config = window.compiler.compile()
    
    print(f"[TEST] Compiled Config: {config}")
    
    # Check params included
    assert config['parameters']['beta'] == 0.95, "Compiled config missing updated parameter."
    assert config['parameters']['phi'] == 1.5, "Compiled config missing second node parameter."
    
    print("[SUCCESS] All Advanced GUI Tests Passed.")

if __name__ == "__main__":
    try:
        test_gui_advanced()
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
