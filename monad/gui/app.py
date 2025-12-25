"""
Monad Studio - Visual Builder
Main application with modern UI/UX design
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QToolBar, QDockWidget, QFormLayout, 
    QDoubleSpinBox, QLabel, QScrollArea, QLineEdit, QComboBox, 
    QCheckBox, QMessageBox, QFrame, QSizePolicy, QSpacerItem,
    QStatusBar, QMenu, QMenuBar, QToolButton
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QAction, QKeySequence, QWheelEvent

from .scene import NodeScene
from .nodes import BlockNode
from .styles import (get_main_stylesheet, get_property_panel_stylesheet, 
                     Colors, Fonts, Icons)
from ..model import MonadModel


class PropertyCard(QFrame):
    """A styled card for grouping properties."""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setObjectName("propertyCard")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
        
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("sectionHeader")
            title_label.setFont(Fonts.small())
            self.layout.addWidget(title_label)


class PropertyEditor(QWidget):
    """Enhanced property editor with card-based layout."""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("propertyPanel")
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)
        
        # Scroll area for content
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        
        self.scroll.setWidget(self.content)
        self.main_layout.addWidget(self.scroll)
        
        self.current_node = None
        self.widgets = {}
        
        # Show empty state
        self._show_empty_state()

    def _show_empty_state(self):
        """Show message when no node is selected."""
        self._clear_content()
        
        empty_label = QLabel("Select a node to edit its properties")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            padding: 40px 20px;
            font-size: 11px;
        """)
        self.content_layout.addWidget(empty_label)
        self.content_layout.addStretch()

    def _clear_content(self):
        """Clear all content from the editor."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.widgets = {}

    def set_node(self, node):
        self.current_node = node
        self._clear_content()
        
        if not node:
            self._show_empty_state()
            return
        
        # Node info card
        info_card = PropertyCard()
        
        # Title with icon
        type_icons = {
            "household": "👥",
            "policy": "🏛",
            "market": "📊",
            "fiscal": "💰",
        }
        icon = type_icons.get(node.node_type, "◆")
        
        title_label = QLabel(f"{icon} {node.title}")
        title_label.setObjectName("nodeTitle")
        title_label.setFont(Fonts.title())
        info_card.layout.addWidget(title_label)
        
        type_label = QLabel(node.node_type.upper())
        type_label.setObjectName("nodeType")
        type_label.setFont(Fonts.small())
        info_card.layout.addWidget(type_label)
        
        self.content_layout.addWidget(info_card)
        
        # Parameters card
        if node.params:
            params_card = PropertyCard("PARAMETERS")
            
            form_layout = QFormLayout()
            form_layout.setSpacing(12)
            form_layout.setLabelAlignment(Qt.AlignRight)
            
            for key, val in node.params.items():
                widget = self._create_widget_for_value(key, val)
                if widget:
                    label = QLabel(f"{key}:")
                    label.setFont(Fonts.body())
                    form_layout.addRow(label, widget)
                    self.widgets[key] = widget
            
            params_card.layout.addLayout(form_layout)
            self.content_layout.addWidget(params_card)
        
        # Actions card
        actions_card = PropertyCard("ACTIONS")
        
        delete_btn = QPushButton("Delete Node")
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR};
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        delete_btn.clicked.connect(self._delete_current_node)
        actions_card.layout.addWidget(delete_btn)
        
        self.content_layout.addWidget(actions_card)
        
        # Spacer
        self.content_layout.addStretch()

    def _create_widget_for_value(self, key, val):
        """Create appropriate widget based on value type."""
        if isinstance(val, bool):
            widget = QCheckBox()
            widget.setChecked(val)
            widget.stateChanged.connect(lambda v, k=key: self.update_param(k, bool(v)))
        elif isinstance(val, float) or isinstance(val, int):
            widget = QDoubleSpinBox()
            widget.setRange(-10000, 10000)
            widget.setSingleStep(0.01)
            widget.setDecimals(4)
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
        
        return widget

    def update_param(self, key, value):
        if self.current_node:
            self.current_node.params[key] = value
    
    def _delete_current_node(self):
        if self.current_node and self.current_node.scene():
            self.current_node.scene().remove_node(self.current_node)
            self.set_node(None)


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with mouse wheel zoom."""
    
    def __init__(self, scene):
        super().__init__(scene)
        self.zoom_factor = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 3.0
        
    def wheelEvent(self, event: QWheelEvent):
        # Zoom with Ctrl + Wheel
        if event.modifiers() == Qt.ControlModifier:
            factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            new_zoom = self.zoom_factor * factor
            
            if self.min_zoom <= new_zoom <= self.max_zoom:
                self.zoom_factor = new_zoom
                self.scale(factor, factor)
            
            event.accept()
        else:
            super().wheelEvent(event)


class NodeEditor(QMainWindow):
    """Main application window with modern UI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monad Studio: Visual Builder")
        self.resize(1400, 900)
        
        # Apply stylesheet
        self.setStyleSheet(get_main_stylesheet() + get_property_panel_stylesheet())
        
        # Central Canvas
        self.scene = NodeScene()
        self.scene.selectionChanged.connect(self.on_selection_changed)
        
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.setCentralWidget(self.view)
        
        # Menu Bar
        self._create_menu_bar()
        
        # Toolbar
        self._create_toolbar()
        
        # Property Dock
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_dock.setMinimumWidth(280)
        self.prop_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.prop_editor = PropertyEditor()
        self.prop_dock.setWidget(self.prop_editor)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status_bar()
        
        # Compiler
        from .compiler import GraphCompiler
        self.compiler = GraphCompiler(self.scene)
        
        # Track changes for status bar
        self.scene.changed.connect(self._update_status_bar)
        
        # Initial Demo Setup
        self.setup_demo()
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_graph)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        toggle_grid = QAction("Toggle Grid", self)
        toggle_grid.setShortcut("G")
        toggle_grid.triggered.connect(self.toggle_grid)
        view_menu.addAction(toggle_grid)
        
        view_menu.addSeparator()
        
        fit_action = QAction("Fit to View", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.fit_to_view)
        view_menu.addAction(fit_action)
        
        reset_zoom = QAction("Reset Zoom", self)
        reset_zoom.setShortcut("0")
        reset_zoom.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom)

    def _create_toolbar(self):
        """Create the main toolbar with styled buttons."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Node addition buttons
        node_buttons = [
            (f"{Icons.HOUSEHOLD} Household (HANK)", self.add_household_hank, "Add HANK household block"),
            (f"{Icons.HOUSEHOLD} Household (RANK)", self.add_household_rank, "Add RANK household block"),
        ]
        
        for text, slot, tooltip in node_buttons:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
        
        toolbar.addSeparator()
        
        policy_buttons = [
            (f"{Icons.POLICY} Policy (Taylor)", self.add_policy, "Add Taylor Rule policy block"),
            (f"{Icons.FISCAL} Government", self.add_fiscal, "Add fiscal government block"),
        ]
        
        for text, slot, tooltip in policy_buttons:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
        
        toolbar.addSeparator()
        
        market_buttons = [
            (f"{Icons.MARKET} Market (NK)", self.add_market, "Add New Keynesian market block"),
            (f"{Icons.MARKET} Market (RBC)", self.add_market_rbc, "Add RBC market block"),
        ]
        
        for text, slot, tooltip in market_buttons:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)
        
        # Run button (styled differently)
        run_btn = QPushButton(f"{Icons.RUN}  RUN MODEL")
        run_btn.setObjectName("runButton")
        run_btn.setMinimumWidth(120)
        run_btn.setToolTip("Compile and run the model (Ctrl+R)")
        run_btn.clicked.connect(self.run_model)
        toolbar.addWidget(run_btn)

    def _update_status_bar(self):
        """Update status bar with current graph info."""
        num_nodes = len(self.scene.nodes)
        num_wires = sum(len(p.wires) for n in self.scene.nodes for p in n.outputs)
        zoom_pct = int(self.view.zoom_factor * 100)
        
        self.status_bar.showMessage(
            f"Nodes: {num_nodes}  |  Connections: {num_wires}  |  "
            f"Zoom: {zoom_pct}%  |  Press G to toggle grid, Ctrl+Wheel to zoom"
        )

    def run_model(self):
        """Compile the visual graph and run the model."""
        print("\n" + "="*60)
        print("--- Semantic Graph Compiler ---")
        print("="*60)
        
        # Compile with semantic analysis
        result = self.compiler.compile()
        
        # Show compilation results
        if result.errors:
            error_msgs = []
            warning_msgs = []
            for err in result.errors:
                if err.severity == "error":
                    error_msgs.append(f"• {err.node_id}: {err.message}")
                else:
                    warning_msgs.append(f"• {err.node_id}: {err.message}")
            
            if error_msgs:
                QMessageBox.critical(self, "Compilation Errors",
                    "The following errors were found:\n\n" + "\n".join(error_msgs))
                return
            
            if warning_msgs:
                QMessageBox.warning(self, "Compilation Warnings",
                    "The following warnings were found:\n\n" + "\n".join(warning_msgs) +
                    "\n\nProceeding with compilation...")
        
        # Print detailed compilation info
        print(f"\n📊 Compiled Graph Summary:")
        print(f"   Nodes: {len(result.nodes)}")
        print(f"   Bindings: {len(result.bindings)}")
        print(f"   Variables: {len(result.variables)}")
        print(f"   Equations: {len(result.equations)}")
        print(f"   Topology: {' → '.join(result.topology)}")
        
        if result.equations:
            print(f"\n📐 Generated Equations:")
            for var, eq in result.equations.items():
                print(f"   {var} := {eq}")
        
        if result.bindings:
            print(f"\n🔗 Variable Bindings:")
            for b in result.bindings:
                print(f"   {b.source_node}.{b.source_var} → {b.target_node}.{b.target_var}")
        
        print("\n" + "="*60)
        
        # Get config for MonadModel
        config = result.to_config()
        
        try:
            m = MonadModel("MonadEngine.exe")
            print("[Visual Builder] Running Monad Engine...")
            try:
                res = m.run(params=config['parameters'])
                
                # Success message with details
                success_msg = (
                    f"Model Solved Successfully!\n\n"
                    f"Nodes: {len(result.nodes)}\n"
                    f"Equations: {len(result.equations)}\n"
                    f"Variables: {len(result.variables)}"
                )
                QMessageBox.information(self, "Success", success_msg)
                
            except Exception as inner_e:
                QMessageBox.warning(self, "Engine Warning", 
                    f"Engine ran but Solver failed or cached data used.\nDetails: {inner_e}")
                
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", str(e))
            print(f"[FAIL] {e}")

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        if items and isinstance(items[0], BlockNode):
            self.prop_editor.set_node(items[0])
        else:
            self.prop_editor.set_node(None)
        self._update_status_bar()

    def add_node(self, title, inputs, outputs, pos, node_type="generic", params=None):
        node = BlockNode(title, inputs, outputs, pos, node_type, params)
        self.scene.add_node(node)
        self._update_status_bar()
        return node

    # --- Node Presets ---
    def add_household_hank(self):
        params = {'agent_type': 'HANK', 'eis': 0.5, 'frisch': 1.0, 'beta': 0.98, 'crypto': False}
        self.add_node("Household", ["Wage (w)", "Rate (r)", "Divs (d)"], 
                     ["Consumption (C)", "Labor (N)"], (100, 300), "household", params)

    def add_household_rank(self):
        params = {'agent_type': 'RANK', 'eis': 0.5, 'frisch': 1.0, 'beta': 0.99, 'crypto': False}
        self.add_node("Household (Rep)", ["Wage (w)", "Rate (r)", "Divs (d)"], 
                     ["Consumption (C)", "Labor (N)"], (100, 300), "household", params)
        
    def add_policy(self):
        params = {'phi_pi': 1.5, 'rho': 0.8, 'r_star': 0.005, 'zlb': True}
        self.add_node("Central Bank", ["Inflation (pi)"], 
                     ["Rate (i)"], (400, 100), "policy", params)

    def add_fiscal(self):
        params = {'G_bar': 0.2, 'B_target': 1.0, 'tax_rate': 0.3}
        self.add_node("Government", ["Output (Y)"], 
                     ["Spending (G)", "Bonds (B)"], (250, 450), "fiscal", params)

    def add_market(self):
        params = {'kappa': 0.1, 'mu': 1.2, 'open_economy': False}
        self.add_node("Goods Market (NK)", ["Demand (C)", "Supply (Y)"], 
                     ["Inflation (pi)", "Divs (d)"], (400, 300), "market", params)

    def add_market_rbc(self):
        params = {'kappa': 1000.0, 'mu': 1.2, 'open_economy': False}
        self.add_node("Goods Market (RBC)", ["Demand (C)", "Supply (Y)"], 
                     ["Inflation (pi)", "Divs (d)"], (400, 300), "market", params)

    # --- View Actions ---
    def new_graph(self):
        """Clear the graph."""
        for node in self.scene.nodes[:]:
            self.scene.remove_node(node)
        self._update_status_bar()
    
    def delete_selected(self):
        """Delete selected nodes."""
        for item in self.scene.selectedItems():
            if isinstance(item, BlockNode):
                self.scene.remove_node(item)
        self.prop_editor.set_node(None)
        self._update_status_bar()
    
    def toggle_grid(self):
        """Toggle grid visibility."""
        self.scene.draw_grid = not self.scene.draw_grid
        self.scene.update()
    
    def fit_to_view(self):
        """Fit all nodes in view."""
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        # Update zoom factor
        transform = self.view.transform()
        self.view.zoom_factor = transform.m11()
        self._update_status_bar()
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.view.resetTransform()
        self.view.zoom_factor = 1.0
        self._update_status_bar()

    def setup_demo(self):
        """Create a standard NK loop for demo."""
        h = self.add_node("Household", ["w", "r", "div"], ["C", "N"], 
                         (150, 350), "household", 
                         {'agent_type': 'HANK', 'eis': 0.5, 'beta': 0.986})
        f = self.add_node("Firm (NK)", ["C (Demand)", "N (Labor)"], 
                         ["w", "pi", "div"], (500, 350), "market", 
                         {'kappa': 0.1})
        cb = self.add_node("Central Bank", ["pi"], ["r"], 
                          (350, 120), "policy", 
                          {'phi_pi': 1.5, 'zlb': True})


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NodeEditor()
    window.show()
    sys.exit(app.exec())
