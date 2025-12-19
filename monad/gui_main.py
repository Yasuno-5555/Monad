import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox, QFormLayout, QMessageBox, QSpinBox,
    QCheckBox, QComboBox, QTabWidget, QSplitter, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal

# Ensure local import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from monad.api import Model
from monad.dsl import AR1

# --- Worker Thread for Heavy Computations ---
class EngineWorker(QThread):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, model, experiment_config):
        super().__init__()
        self.model = model
        self.config = experiment_config

    def run(self):
        try:
            # 1. Initialize (Cached)
            self.progress.emit("Initializing Model & Caching...")
            self.model.initialize()
            
            # 2. Configure Experiment
            self.progress.emit("Running Simulation...")
            shock_type = self.config['type']
            T = self.model.T
            
            shocks = {}
            if shock_type == 'monetary':
                # Natural Rate Shock
                val = self.config['shock_val']
                persistence = self.config['persistence']
                shocks['dr_star'] = val * AR1(persistence, 1.0, T)
                zlb = self.config['zlb']
                results = self.model.run_experiment(shocks, zlb=zlb, robust=True)
                
            elif shock_type == 'fiscal':
                # Fiscal Shock
                val_G = self.config['dG']
                val_T = self.config['dTrans']
                persistence = self.config['persistence']
                shocks['dG'] = val_G * AR1(persistence, 1.0, T)
                shocks['dTrans'] = val_T * AR1(persistence, 1.0, T)
                # Fiscal runs usually without ZLB logic in current simple backend, or partial eq
                # But Model.run_experiment handles dispatch
                results = self.model.run_experiment(shocks, zlb=False) # Simplified for fiscal

            # 3. Post-Process / Analysis
            self.progress.emit("Analyzing Results...")
            
            # Inequality Analysis
            if 'dr' in results and 'Y' in results: # Y as proxy for Z
                ineq = self.model.analyze_inequality(results)
                results['analysis_ineq'] = ineq
            
            # Decomposition (if fiscal/monetary)
            # Need dr, dY, dTrans
            dr_path = results.get('dr', np.zeros(T))
            dY_path = results.get('dY', results.get('Y', np.zeros(T)))
            dTrans_path = shocks.get('dTrans', np.zeros(T))
            
            decomp = self.model.backend.decompose_multiplier(dY_path, dTrans_path, dr_path)
            results['analysis_decomp'] = decomp
            
            # MPC Stats
            mpc = self.model.backend.compute_mpc_distribution()
            results['analysis_mpc'] = mpc

            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class DashboardCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(DashboardCanvas, self).__init__(self.fig)
        self.setParent(parent)

class MonadCockpit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monad Studio v2.3 - Advanced Workbench")
        self.setMinimumSize(1400, 900)
        
        # Determine paths relative to script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, ".monad_cache")

        # Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === Left Sidebar (Controls) ===
        sidebar = QWidget()
        sidebar.setFixedWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # 1. Model Configuration
        grp_model = QGroupBox("Model Configuration")
        form_model = QFormLayout(grp_model)
        
        self.combo_model = QComboBox()
        self.combo_model.addItem("Two-Asset HANK", "two_asset")
        self.combo_model.addItem("One-Asset HANK", "one_asset")
        self.combo_model.addItem("Representative Agent (RANK)", "rank")
        form_model.addRow("Model Type:", self.combo_model)
        
        self.check_sticky = QCheckBox("Sticky Prices (NKPC)")
        self.check_sticky.setChecked(True)
        form_model.addRow(self.check_sticky)
        
        sidebar_layout.addWidget(grp_model)
        
        # 2. Parameters
        grp_param = QGroupBox("Structural Parameters")
        form_param = QFormLayout(grp_param)
        
        self.spin_alpha = self._make_spin(0.30, 0, 1, 0.05)
        form_param.addRow("Import Share (α):", self.spin_alpha)
        
        self.spin_chi = self._make_spin(1.0, 0, 5, 0.1)
        form_param.addRow("Trade Elast. (χ):", self.spin_chi)
        
        sidebar_layout.addWidget(grp_param)
        
        # 3. Experiment Settings (Tabs)
        grp_exp = QGroupBox("Experiment Design")
        exp_layout = QVBoxLayout(grp_exp)
        self.tab_exp = QTabWidget()
        
        # Tab 1: Monetary Policy
        tab_mon = QWidget()
        form_mon = QFormLayout(tab_mon)
        self.spin_r_shock = self._make_spin(-0.02, -0.1, 0.1, 0.005, dec=3)
        form_mon.addRow("r* Shock Size:", self.spin_r_shock)
        self.spin_r_pers = self._make_spin(0.9, 0, 1, 0.05)
        form_mon.addRow("Persistence:", self.spin_r_pers)
        self.check_zlb = QCheckBox("Enable ZLB")
        self.check_zlb.setChecked(True)
        form_mon.addRow(self.check_zlb)
        self.tab_exp.addTab(tab_mon, "Monetary")
        
        # Tab 2: Fiscal Policy
        tab_fis = QWidget()
        form_fis = QFormLayout(tab_fis)
        self.spin_g_shock = self._make_spin(0.01, -0.1, 0.1, 0.005, dec=3)
        form_fis.addRow("Avg Spending (dG):", self.spin_g_shock)
        self.spin_t_shock = self._make_spin(0.00, -0.1, 0.1, 0.005, dec=3)
        form_fis.addRow("Transfer (dT):", self.spin_t_shock)
        self.spin_f_pers = self._make_spin(0.9, 0, 1, 0.05)
        form_fis.addRow("Persistence:", self.spin_f_pers)
        self.tab_exp.addTab(tab_fis, "Fiscal")
        
        exp_layout.addWidget(self.tab_exp)
        sidebar_layout.addWidget(grp_exp)
        
        # Run Button
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.start_simulation)
        sidebar_layout.addWidget(self.btn_run)
        
        # Status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: gray;")
        sidebar_layout.addWidget(self.lbl_status)
        
        sidebar_layout.addStretch()
        
        # === Main Content (Analysis Dashboard) ===
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)
        
        self.tab_analysis = QTabWidget()
        
        # Tab 1: Aggregate (IRFs)
        self.canvas_agg = DashboardCanvas(self)
        self.tab_analysis.addTab(self.canvas_agg, "Aggregate Macros")
        
        # Tab 2: Inequality (Heatmap & Groups)
        self.canvas_ineq = DashboardCanvas(self)
        self.tab_analysis.addTab(self.canvas_ineq, "Inequality Analysis")
        
        # Tab 3: Mechanisms (Decomposition & MPC)
        self.canvas_mech = DashboardCanvas(self)
        self.tab_analysis.addTab(self.canvas_mech, "Mechanisms")
        
        content_layout.addWidget(self.tab_analysis)
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(main_content)
        
        # Worker placeholder
        self.worker = None

    def _make_spin(self, val, min_val, max_val, step, dec=2):
        s = QDoubleSpinBox()
        s.setRange(min_val, max_val)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setValue(val)
        return s

    def start_simulation(self):
        # Disable button
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Preparing...")
        self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
        
        # Config
        model_type = self.combo_model.currentData()
        
        # Param Logic
        alpha = self.spin_alpha.value()
        chi = self.spin_chi.value()
        is_sticky = self.check_sticky.isChecked()
        kappa = 0.1 if is_sticky else 100.0
        
        params = {'alpha': alpha, 'chi': chi, 'kappa': kappa, 'beta': 0.99, 'phi_pi': 1.5}
        
        # Experiment Logic
        exp_idx = self.tab_exp.currentIndex()
        if exp_idx == 0: # Monetary
            exp_config = {
                'type': 'monetary',
                'shock_val': self.spin_r_shock.value(),
                'persistence': self.spin_r_pers.value(),
                'zlb': self.check_zlb.isChecked()
            }
        else: # Fiscal
            exp_config = {
                'type': 'fiscal',
                'dG': self.spin_g_shock.value(),
                'dTrans': self.spin_t_shock.value(),
                'persistence': self.spin_f_pers.value()
            }
        
        # Initialize Model Wrapper
        model = Model(model_type=model_type, T=50, params=params, cache_dir=self.cache_dir)
        
        # Threading
        self.worker = EngineWorker(model, exp_config)
        self.worker.progress.connect(lambda s: self.lbl_status.setText(s))
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Error")
        self.lbl_status.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Error", msg)

    def on_finished(self, results):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Complete")
        self.lbl_status.setStyleSheet("color: green;")
        
        # Plotting
        self.plot_aggregate(results)
        self.plot_inequality(results)
        self.plot_mechanisms(results)

    def plot_aggregate(self, res):
        canvas = self.canvas_agg
        canvas.fig.clear()
        
        vars = ['Y', 'C_agg', 'i', 'pi']
        titles = ['GDP', 'Consumption', 'Nominal Rate', 'Inflation']
        
        axes = canvas.fig.subplots(2, 2)
        axes = axes.flatten()
        
        t = np.arange(len(res.get('Y', [])))
        
        for ax, var, title in zip(axes, vars, titles):
            if var in res:
                y = res[var]
                if np.max(np.abs(y)) < 0.2: y = y * 100
                ax.plot(t, y, lw=2, color='#2c3e50')
                ax.axhline(0, c='gray', ls='--', lw=0.5)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
        canvas.fig.tight_layout()
        canvas.draw()

    def plot_inequality(self, res):
        canvas = self.canvas_ineq
        canvas.fig.clear()
        
        if 'analysis_ineq' not in res:
            canvas.fig.text(0.5, 0.5, "No Inequality Data", ha='center')
            canvas.draw()
            return
            
        ineq = res['analysis_ineq']
        t = np.arange(len(ineq['top10']))
        
        # Left: Group Response
        ax1 = canvas.fig.add_subplot(121)
        ax1.plot(t, ineq['top10']*100, label='Top 10%', color='#e74c3c')
        ax1.plot(t, ineq['bottom50']*100, label='Bottom 50%', color='#3498db')
        ax1.plot(t, ineq['debtors']*100, label='Debtors', color='#9b59b6', ls='--')
        ax1.set_title("Consumption by Group (% Dev)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, c='k', lw=0.5)
        
        # Right: MPC Histogram (from Micro)
        # Or Heatmap? Heatmap is flattened [im, ia, iz], requires reshaping.
        # Let's plot MPC stats instead if Heatmap is hard to reshape without grid dims
        
        if 'analysis_mpc' in res:
            ax2 = canvas.fig.add_subplot(122)
            mpc = res['analysis_mpc']
            # Bar chart of MPC by Income State
            mpc_z = mpc.get('mpc_by_z', [])
            x = np.arange(len(mpc_z))
            ax2.bar(x, mpc_z, color='teal', alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Income {i+1}" for i in x])
            ax2.set_title("Avg MPC by Income State")
            ax2.set_ylim(0, 1.0)
            
            # Text Stats
            avg = mpc.get('weighted_mpc', 0)
            ax2.text(0.95, 0.95, f"Agg MPC: {avg:.2f}", transform=ax2.transAxes, ha='right')

        canvas.fig.tight_layout()
        canvas.draw()

    def plot_mechanisms(self, res):
        canvas = self.canvas_mech
        canvas.fig.clear()
        
        if 'analysis_decomp' not in res:
            canvas.fig.text(0.5, 0.5, "No Decomposition Data", ha='center')
            canvas.draw()
            return
            
        decomp = res['analysis_decomp']
        top = decomp['direct']
        bottom = decomp['indirect']
        t = np.arange(len(top))
        
        ax = canvas.fig.add_subplot(111)
        ax.bar(t, top, label='Direct Effect (Partial Eq)', color='#f1c40f', alpha=0.8)
        ax.bar(t, bottom, bottom=top, label='Indirect Effect (GE Multiplier)', color='#e67e22', alpha=0.8)
        
        total = top + bottom
        ax.plot(t, total, 'k--', label='Total Effect', lw=2)
        
        ax.set_title("GDP Multiplier Decomposition")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas.fig.tight_layout()
        canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Optional: Modern Style
    app.setStyle("Fusion")
    
    window = MonadCockpit()
    window.show()
    sys.exit(app.exec())
