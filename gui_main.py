import sys
import os
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox, QFormLayout, QMessageBox, QSpinBox,
    QCheckBox, QComboBox
)
from PySide6.QtCore import Qt

# --- Matplotlib Integration ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# --- Monad Engine Integration ---
# Ensure we can import from the local package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from monad.solver import SOESolver
from monad.nonlinear import NewtonSolver

class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib Canvas Widget"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # Create 2 subplots (Top: Macro, Bottom: Policy/Prices)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        # Adjust spacing
        self.fig.subplots_adjust(hspace=0.4, bottom=0.1, top=0.95)
        super(MplCanvas, self).__init__(self.fig)

class MonadCockpit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monad Engine v6.0 - Control Deck")
        self.setMinimumSize(1200, 900)

        # --- 1. Load Engine Data (Once) ---
        # Assume CSVs are in the current directory or data folder
        # Adjust path if necessary!
        self.path_R = "gpu_jacobian_R.csv" 
        self.path_Z = "gpu_jacobian_Z.csv"
        
        if not os.path.exists(self.path_R):
            print("WARNING: GPU CSVs not found in root. Please generate them first.")

        # --- GUI Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === Left Panel: Controls ===
        control_panel = QGroupBox("Simulation Parameters")
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)

        # Form Layout for Inputs
        form_layout = QFormLayout()

        # Input: Import Share (Alpha)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.30) # Default: Japan-like
        form_layout.addRow("Import Share (α):", self.spin_alpha)

        # Input: Export Elasticity (Chi)
        self.spin_chi = QDoubleSpinBox()
        self.spin_chi.setRange(0.0, 5.0)
        self.spin_chi.setSingleStep(0.1)
        self.spin_chi.setValue(0.20) # Default: Low elasticity (Paradox zone)
        form_layout.addRow("Export Elast. (χ):", self.spin_chi)
        
        # Input: Shock Size (r* drop)
        self.spin_shock = QDoubleSpinBox()
        self.spin_shock.setRange(-0.10, 0.0)
        self.spin_shock.setSingleStep(0.005)
        self.spin_shock.setDecimals(3)
        self.spin_shock.setValue(-0.020) # -2% shock
        form_layout.addRow("Natural Rate Shock:", self.spin_shock)

        control_layout.addLayout(form_layout)
        
        control_layout.addSpacing(10)
        
        # === Model Settings ===
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout(model_group)
        
        # Model Type Selector
        self.combo_model = QComboBox()
        self.combo_model.addItem("Two-Asset HANK (Heterogeneous)", "two_asset")
        self.combo_model.addItem("One-Asset RANK (Representative)", "one_asset")
        model_layout.addRow("Model Type:", self.combo_model)
        
        # ZLB Toggle
        self.check_zlb = QCheckBox("Enable Zero Lower Bound")
        self.check_zlb.setChecked(True)  # Default: ZLB ON
        self.check_zlb.setStyleSheet("font-weight: bold;")
        model_layout.addRow(self.check_zlb)
        
        control_layout.addWidget(model_group)
        
        control_layout.addSpacing(10)
        
        # Run Button
        self.run_button = QPushButton("Run Simulation (ZLB)")
        self.run_button.setFixedHeight(60)
        self.run_button.setStyleSheet("""
            QPushButton { 
                background-color: #d35400; color: white; 
                font-weight: bold; font-size: 16px; border-radius: 8px; 
            }
            QPushButton:hover { background-color: #e67e22; }
            QPushButton:pressed { background-color: #a04000; }
        """)
        self.run_button.clicked.connect(self.run_engine)
        
        control_layout.addWidget(self.run_button)
        
        # Error Label (For convergence warnings)
        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #c0392b; font-weight: bold; font-size: 14px; margin-top: 5px;")
        control_layout.addWidget(self.error_label)

        control_layout.addSpacing(10)
        
        # === Advanced Settings (Collapsible) ===
        self.advanced_group = QGroupBox("⚙ Advanced Solver Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)  # Collapsed by default
        self.advanced_group.setStyleSheet("""
            QGroupBox::indicator { width: 13px; height: 13px; }
            QGroupBox::indicator:unchecked { background-color: #bdc3c7; border-radius: 2px; }
            QGroupBox::indicator:checked { background-color: #27ae60; border-radius: 2px; }
        """)
        advanced_layout = QFormLayout(self.advanced_group)
        
        # Max Iterations
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(100, 10000)
        self.spin_max_iter.setSingleStep(100)
        self.spin_max_iter.setValue(1000)
        advanced_layout.addRow("Max Iterations:", self.spin_max_iter)
        
        # Damping Factor
        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(0.01, 1.0)
        self.spin_damping.setSingleStep(0.05)
        self.spin_damping.setDecimals(2)
        self.spin_damping.setValue(0.20)
        advanced_layout.addRow("Damping Factor:", self.spin_damping)
        
        # Homotopy Steps
        self.spin_homotopy_steps = QSpinBox()
        self.spin_homotopy_steps.setRange(3, 20)
        self.spin_homotopy_steps.setSingleStep(1)
        self.spin_homotopy_steps.setValue(5)
        advanced_layout.addRow("Homotopy Steps:", self.spin_homotopy_steps)
        
        control_layout.addWidget(self.advanced_group)

        control_layout.addStretch()
        
        # === Right Panel: Dashboard ===
        dashboard_panel = QGroupBox("Real-time Analysis")
        dashboard_layout = QVBoxLayout(dashboard_panel)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        dashboard_layout.addWidget(self.canvas)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(dashboard_panel)

        self.statusBar().showMessage("Engine Online. Awaiting Orders.")

    def run_engine(self):
        """Connects GUI Inputs -> Monad Engine -> Matplotlib"""
        self.statusBar().showMessage("Computing Nonlinear Equilibrium...")
        self.error_label.setText("") # Clear previous errors
        QApplication.processEvents() # Force UI update

        try:
            # 1. Get Parameters from GUI
            alpha = self.spin_alpha.value()
            chi = self.spin_chi.value()
            shock_val = self.spin_shock.value()
            
            # Model Settings
            model_type = self.combo_model.currentData()
            zlb_enabled = self.check_zlb.isChecked()

            # 2. Setup Solver (On the fly!)
            # Start with a low chi for linear backbone (will be updated by homotopy if needed)
            base_chi = min(chi, 0.2)  # Safe starting point
            params = {'alpha': alpha, 'chi': base_chi, 'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
            
            # Linear Backbone
            soe_solver = SOESolver(self.path_R, self.path_Z, T=50, params=params)
            
            # Get advanced settings (use defaults if panel is collapsed)
            if self.advanced_group.isChecked():
                max_iter = self.spin_max_iter.value()
                damping = self.spin_damping.value()
                homotopy_steps = self.spin_homotopy_steps.value()
            else:
                max_iter = 1000
                damping = 0.2
                homotopy_steps = 5
            
            # Nonlinear Wrapper
            solver = NewtonSolver(soe_solver, max_iter=max_iter, tol=1e-6)
            solver.damping = damping  # Override damping
            solver.zlb_enabled = zlb_enabled  # Pass ZLB setting
            solver.model_type = model_type  # Pass model type

            # 3. Define Shock (Persistent r* drop)
            rho = 0.9
            shock_path = shock_val * (rho ** np.arange(50))

            # 4. Solve! (Use Homotopy for high chi values)
            HOMOTOPY_THRESHOLD = 0.3
            if chi >= HOMOTOPY_THRESHOLD:
                # High chi -> use homotopy continuation
                steps = homotopy_steps if self.advanced_group.isChecked() else max(3, int((chi - base_chi) / 0.1) + 1)
                zlb_str = "ZLB" if zlb_enabled else "No ZLB"
                self.statusBar().showMessage(f"Using Homotopy (χ={chi}, {steps} steps, {zlb_str})...")
                QApplication.processEvents()
                results = solver.solve_with_homotopy(shock_path=shock_path, target_chi=chi, steps=steps)
            else:
                # Low chi -> direct solve
                results = solver.solve_nonlinear(shock_path=shock_path)

            # 5. Visualize
            self.plot_results(results)
            zlb_str = "ZLB" if zlb_enabled else "No ZLB"
            model_str = "HANK" if model_type == "two_asset" else "RANK"
            self.statusBar().showMessage(f"Complete. (α={alpha}, χ={chi}, {model_str}, {zlb_str})")

        except RuntimeError as e:
            # Specific handling for convergence failure
            if "converge" in str(e).lower():
                self.error_label.setText("⚠ CONVERGENCE FAILED")
                self.statusBar().showMessage("Simulation Failed: Solver did not converge.")
                QMessageBox.critical(self, "Solver Error", 
                    f"Convergence Failed!\n\nThe engine could not find an equilibrium.\nHint: Try lowering the shock size or changing parameters.\n\nDetails: {str(e)}")
            else:
                self.error_label.setText("⚠ RUNTIME ERROR")
                self.statusBar().showMessage(f"Error: {str(e)}")
                QMessageBox.critical(self, "Runtime Error", f"An error occurred during simulation.\n\nDetails: {str(e)}")
            print(e)
            
        except Exception as e:
            self.error_label.setText("⚠ ERROR")
            self.statusBar().showMessage(f"Error: {str(e)}")
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred.\n\nDetails: {str(e)}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()

    def plot_results(self, res):
        """Update the Matplotlib Canvas"""
        t = np.arange(len(res['Y']))
        
        # --- Ax1: Real Economy (GDP vs Cons) ---
        self.canvas.ax1.cla()
        self.canvas.ax1.plot(t, res['Y']*100, label='GDP (Y)', color='blue', lw=2)
        self.canvas.ax1.plot(t, res['C_agg']*100, label='Consumption (C)', color='red', lw=2, ls='--')
        self.canvas.ax1.axhline(0, color='black', lw=0.5)
        self.canvas.ax1.set_title("Real Economy: The Disconnect")
        self.canvas.ax1.set_ylabel("% Deviation")
        self.canvas.ax1.legend()
        self.canvas.ax1.grid(True, alpha=0.3)

        # --- Ax2: Policy & Prices ---
        # Need to reconstruct Q if not in results (depends on implementation)
        # But let's plot Interest Rate and Inflation
        self.canvas.ax2.cla()
        self.canvas.ax2.plot(t, res['i']*100, label='Nominal Rate (i)', color='green', lw=2)
        self.canvas.ax2.plot(t, res['pi']*100, label='Inflation (π)', color='orange', lw=2, ls=':')
        self.canvas.ax2.axhline(0, color='black', lw=0.5)
        self.canvas.ax2.set_title("Nominal Side: Zero Lower Bound")
        self.canvas.ax2.set_ylabel("% / Annual")
        self.canvas.ax2.legend()
        self.canvas.ax2.grid(True, alpha=0.3)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MonadCockpit()
    window.show()
    sys.exit(app.exec())
