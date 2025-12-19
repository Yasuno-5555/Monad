import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QToolBar, QDockWidget, QFormLayout, QDoubleSpinBox, QLabel, QScrollArea, QLineEdit, QComboBox, QCheckBox, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter

from .scene import NodeScene
from .nodes import BlockNode
from ..model import MonadModel

class PropertyEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QFormLayout(self)
        self.current_node = None
        self.widgets = {}

    def set_node(self, node):
        self.current_node = node
        # Clear existing
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        self.widgets = {}
        
        if not node:
            self.layout.addRow(QLabel("No selection"))
            return
            
        self.layout.addRow(QLabel(f"<b>{node.title}</b> ({node.node_type})"))
        
        for key, val in node.params.items():
            # Handle different types
            if isinstance(val, bool):
                widget = QCheckBox()
                widget.setChecked(val)
                # For checkbox, signal is stateChanged (int) or clicked(bool)
                widget.stateChanged.connect(lambda v, k=key: self.update_param(k, bool(v)))
            elif isinstance(val, float) or isinstance(val, int):
                widget = QDoubleSpinBox()
                widget.setRange(-10000, 10000)
                widget.setSingleStep(0.01)
                widget.setValue(float(val))
                widget.valueChanged.connect(lambda v, k=key: self.update_param(k, v))
            elif isinstance(val, str):
                if val in ["HANK", "RANK", "TANK"]:
                    widget = QComboBox()
                    widget.addItems(["HANK", "RANK", "TANK"])
                    widget.setCurrentText(val)
                    widget.currentTextChanged.connect(lambda v, k=key: self.update_param(k, v))
                else:
                    widget = QLineEdit(val)
                    widget.textChanged.connect(lambda v, k=key: self.update_param(k, v))
            else:
                widget = QLabel(str(val))

            self.layout.addRow(f"{key}:", widget)
            self.widgets[key] = widget

    def update_param(self, key, value):
        if self.current_node:
            self.current_node.params[key] = value

class NodeEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monad Studio: Visual Builder")
        self.resize(1200, 800)
        
        # Central Canvas
        self.scene = NodeScene()
        self.scene.selectionChanged.connect(self.on_selection_changed)
        
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag) # Pan with simple click
        
        self.setCentralWidget(self.view)
        
        # Toolbar
        toolbar = QToolBar("Blocks")
        self.addToolBar(toolbar)
        
        # Add Block Actions
        self._add_action(toolbar, "Add Household (HANK)", self.add_household_hank)
        self._add_action(toolbar, "Add Household (RANK)", self.add_household_rank)
        toolbar.addSeparator()
        self._add_action(toolbar, "Add Policy", self.add_policy)
        self._add_action(toolbar, "Add Government (Fiscal)", self.add_fiscal)
        self._add_action(toolbar, "Add Market (NK)", self.add_market)
        self._add_action(toolbar, "Add Market (RBC)", self.add_market_rbc)
        
        toolbar.addSeparator()
        self._add_action(toolbar, "RUN MODEL", self.run_model)
        
        # Property Dock
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_editor = PropertyEditor()
        self.prop_dock.setWidget(self.prop_editor)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)
        
        # Compiler
        from .compiler import GraphCompiler
        self.compiler = GraphCompiler(self.scene)
        
        # Initial Demo Setup
        self.setup_demo()
        
    def _add_action(self, toolbar, name, slot):
        btn = QPushButton(name)
        btn.clicked.connect(slot)
        toolbar.addWidget(btn)

    def run_model(self):
        print("--- Compiling Visual Graph ---")
        config = self.compiler.compile()
        print("Configuration:", config)
        
        try:
            m = MonadModel("MonadEngine.exe")
            # Inject Visual Config into Model
            print("[Visual Builder] Running Monad Engine...")
            try:
                # Assuming MonadModel.run takes params dict.
                res = m.run(params=config['parameters'])
                QMessageBox.information(self, "Success", "Model Solved Successfully.\nResults cached.")
            except Exception as inner_e:
                 # Check if partial results available or just display warning
                QMessageBox.warning(self, "Engine Warning", f"Engine ran but Solver failed or cached data used.\nDetails: {inner_e}")
                
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", str(e))
            print(f"[FAIL] {e}")

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        if items and isinstance(items[0], BlockNode):
            self.prop_editor.set_node(items[0])
        else:
            self.prop_editor.set_node(None)

    def add_node(self, title, inputs, outputs, pos, node_type="generic", params=None):
        node = BlockNode(title, inputs, outputs, pos, node_type, params)
        self.scene.add_node(node)
        return node

    # --- Presets ---
    def add_household_hank(self):
        params = {'agent_type': 'HANK', 'eis': 0.5, 'frisch': 1.0, 'beta': 0.98, 'crypto': False}
        self.add_node("Household", ["Wage (w)", "Rate (r)", "Divs (d)"], ["Consumption (C)", "Labor (N)"], (100, 300), "household", params)

    def add_household_rank(self):
        params = {'agent_type': 'RANK', 'eis': 0.5, 'frisch': 1.0, 'beta': 0.99, 'crypto': False}
        self.add_node("Household (Rep)", ["Wage (w)", "Rate (r)", "Divs (d)"], ["Consumption (C)", "Labor (N)"], (100, 300), "household", params)
        
    def add_policy(self):
        params = {'phi_pi': 1.5, 'rho': 0.8, 'r_star': 0.005, 'zlb': True}
        self.add_node("Central Bank", ["Inflation (pi)"], ["Rate (i)"], (400, 100), "policy", params)

    def add_fiscal(self):
        params = {'G_bar': 0.2, 'B_target': 1.0, 'tax_rate': 0.3}
        self.add_node("Government", ["Output (Y)"], ["Spending (G)", "Bonds (B)"], (250, 450), "fiscal", params)

    def add_market(self):
        params = {'kappa': 0.1, 'mu': 1.2, 'open_economy': False}
        self.add_node("Goods Market (NK)", ["Demand (C)", "Supply (Y)"], ["Inflation (pi)", "Divs (d)"], (400, 300), "market", params)

    def add_market_rbc(self):
        # RBC = Flexible Prices (High Kappa)
        params = {'kappa': 1000.0, 'mu': 1.2, 'open_economy': False}
        self.add_node("Goods Market (RBC)", ["Demand (C)", "Supply (Y)"], ["Inflation (pi)", "Divs (d)"], (400, 300), "market", params)
        
    def setup_demo(self):
        # Create a standard NK loop
        # Default to HANK in demo
        h = self.add_node("Household", ["w", "r", "div"], ["C", "N"], (100, 300), "household", {'agent_type': 'HANK', 'eis': 0.5, 'beta': 0.986})
        f = self.add_node("Firm (New Keynesian)", ["C (Demand)", "N (Labor)"], ["w", "pi", "div"], (400, 300), "market", {'kappa': 0.1})
        cb = self.add_node("Central Bank (Taylor)", ["pi"], ["r"], (250, 100), "policy", {'phi_pi': 1.5, 'zlb': True})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NodeEditor()
    window.show()
    sys.exit(app.exec())
