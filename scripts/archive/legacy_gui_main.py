import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QDoubleSpinBox, QFormLayout, QMessageBox, QSpinBox,
    QCheckBox, QComboBox, QTabWidget, QSplitter, QScrollArea, QFrame, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal

# Ensure local import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from monad.api import Model
from monad.dsl import AR1
from monad.modeling.schema import ModelSpec
from monad.modeling.dsge import DSGEStaticSolver

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
            # Check for Generic Solver
            if hasattr(self.model, 'spec'): # It's a DSGEStaticSolver
                self.progress.emit("Solving Descriptor Model...")
                
                # Setup Boundary Conditions (Simple One-time Shock)
                # We assume we are solving a transition path returning to SS.
                # We shock the initial state of the first variable (often Y) or specific if found.
                
                # 1. Get Steady State (Assume 0 for now or whatever is in spec guesses)
                # Ideally we solve for SS first. 
                # For this demo, we assume the Spec *IS* the SS (e.g. guesses are SS).
                
                # 2. Apply Shock to Initial State
                # Using 'shock_val' from config (Monetary) or 'dG' (Fiscal)
                val = self.config.get('shock_val')
                if val is None: val = self.config.get('dG', 0.0)
                
                # Heuristic: Find a suitable state variable to shock
                # Priority: 'e_r', starts with 'e_', 'eps_', or 'shock'
                target_var = None
                vars = list(self.model.var_names)
                
                # Priority 1: Exact match 'e_r' (Natural Rate Shock)
                if 'e_r' in vars: target_var = 'e_r'
                
                # Priority 2: Exogenous process notation
                if not target_var:
                    for v in vars:
                        if v.startswith('e_') or v.startswith('eps_') or 'shock' in v:
                            target_var = v
                            break
                            
                # Priority 3: First variable
                if not target_var: target_var = vars[0]
                
                print(f"[Engine] Applying shock {val} to variable '{target_var}'")
                
                current_init = self.model.initial_state.copy()
                current_init[target_var] += val
                self.model.set_initial_state(current_init)
                
                # 3. Solve
                res_cols = self.model.solve() # Returns dict {var: np.array}
                
                # 4. Pack results
                results = res_cols
                # Add dummy analysis keys to avoid plot errors
                results['analysis_ineq'] = {
                    'top10': np.zeros(self.model.T), 
                    'bottom50': np.zeros(self.model.T),
                    'debtors': np.zeros(self.model.T)
                }
                
                # Mechanisms: For RANK/NK, Total approx Direct (PE).
                # We map the main solution variable (e.g. y) to 'Direct', leaving Indirect=0.
                proxy_var = 'y' if 'y' in results else list(results.keys())[0]
                results['analysis_decomp'] = {
                    'direct': results[proxy_var], 
                    'indirect': np.zeros(self.model.T)
                }
                
                self.progress.emit("Solved.")
                self.finished.emit(results)
                return

            # --- Legacy Logic ---
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
        
        self.current_spec = None
        self.param_widgets = {} # Map param name -> QDoubleSpinBox
        self.history = [] # For storing multiple run results [(label, res), ...]

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
        
        self.btn_load = QPushButton("Load Model (.yaml)")
        self.btn_load.clicked.connect(self.load_model_yaml)
        form_model.addRow(self.btn_load)
        
        self.combo_model = QComboBox()
        # Legacy
        self.combo_model.addItem("Two-Asset HANK (Legacy)", "two_asset")
        self.combo_model.addItem("One-Asset HANK (Legacy)", "one_asset")
        self.combo_model.addItem("Representative Agent (Legacy)", "rank")
        
        # Presets
        self.preset_files = {}
        presets_dir = os.path.join(base_dir, "presets")
        if os.path.exists(presets_dir):
            for f in os.listdir(presets_dir):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    # Use filename as key, try to read Name from yaml? Too slow? Just filename.
                    name = os.path.splitext(f)[0].replace("_", " ").title()
                    path = os.path.join(presets_dir, f)
                    self.preset_files[name] = path
                    self.combo_model.addItem(f"Preset: {name}", f"preset:{name}")

        self.combo_model.addItem("Custom Descriptor (Active)", "descriptor")
        # Find index of Custom and disable initially?
        idx_custom = self.combo_model.findData("descriptor")
        self.combo_model.model().item(idx_custom).setEnabled(False)

        form_model.addRow("Model Type:", self.combo_model)
        
        # Description Label
        self.lbl_desc = QLabel("Select a model preset or load custom file.")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("font-style: italic; color: #666; padding: 5px; background: #f0f0f0; border-radius: 4px;")
        form_model.addRow(self.lbl_desc)
        
        self.check_sticky = QCheckBox("Sticky Prices (NKPC)")
        self.check_sticky.setChecked(True)
        form_model.addRow(self.check_sticky)
        
        sidebar_layout.addWidget(grp_model)
        
        # 2. Parameters
        self.grp_param = QGroupBox("Structural Parameters") # Store ref to enable/disable
        self.form_param = QFormLayout(self.grp_param)
        sidebar_layout.addWidget(self.grp_param)
        
        # Connect Combo *after* UI setup
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        
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
        exp_layout.addWidget(self.tab_exp)
        sidebar_layout.addWidget(grp_exp)
        
        # 4. Advanced Computation (Collapsible)
        self.grp_adv = QGroupBox("Advanced Computation Settings")
        self.grp_adv.setCheckable(True)
        self.grp_adv.setChecked(False) # Collapsed by default
        adv_layout = QFormLayout(self.grp_adv)
        
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 10000)
        self.spin_iter.setValue(200)
        self.spin_iter.setSingleStep(50)
        adv_layout.addRow("Max Iterations:", self.spin_iter)
        
        self.spin_damp = self._make_spin(0.5, 0.01, 1.0, 0.05)
        adv_layout.addRow("Damping Factor:", self.spin_damp)
        
        self.spin_tol = self._make_spin(1e-6, 1e-9, 1e-2, 1e-7, dec=9)
        adv_layout.addRow("Tolerance:", self.spin_tol)
        
        sidebar_layout.addWidget(self.grp_adv)
        
        # Run & Clear Buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.start_simulation)
        btn_layout.addWidget(self.btn_run, 3)
        
        self.btn_clear = QPushButton("Clear\nPlots")
        self.btn_clear.setFixedHeight(50)
        self.btn_clear.clicked.connect(self.clear_plots)
        btn_layout.addWidget(self.btn_clear, 1)
        
        self.btn_export = QPushButton("Export\nFigure")
        self.btn_export.setFixedHeight(50)
        self.btn_export.clicked.connect(self.export_plots)
        btn_layout.addWidget(self.btn_export, 1)
        
        sidebar_layout.addLayout(btn_layout)
        
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
        
        # Auto-Load Default Model
        default_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'nk_model.yaml')
        if os.path.exists(default_yaml):
            self.load_model_yaml(default_yaml)

    def _load_spec_from_path(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.current_spec = ModelSpec.from_yaml(content)
            
            # Update Description
            desc = self.current_spec.description
            if not desc: desc = "No description available."
            self.lbl_desc.setText(desc)
            
            return True
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return False

    def load_model_yaml(self, path=None):
        # Button handler or explicit load
        fname = path
        if fname is None:
            fname, _ = QFileDialog.getOpenFileName(self, "Open Model Descriptor", "", "YAML Files (*.yaml *.yml)")
            if not fname: return

        if self._load_spec_from_path(fname):
            self.statusBar().showMessage(f"Loaded Custom: {self.current_spec.name}")
            
            # Enable Custom Item
            idx = self.combo_model.findData("descriptor")
            if idx >= 0:
                self.combo_model.model().item(idx).setEnabled(True)
                
            # Switch to Custom (triggers on_model_changed)
            # Use blockSignals to avoid double-loading if we want, but simple switch is fine
            self.combo_model.setCurrentIndex(idx)
            
            if path is None: # Only show dialog success for manual load
                QMessageBox.information(self, "Success", f"Model '{self.current_spec.name}' loaded.")

    def on_model_changed(self, index):
        data = str(self.combo_model.itemData(index))
        self.grp_param.setEnabled(True) # Default unlock
        
        if data.startswith("preset:"):
            name = data.replace("preset:", "")
            path = self.preset_files.get(name)
            if path and self._load_spec_from_path(path):
                self.setup_descriptor_ui()
                self.grp_param.setEnabled(False) # LOCK for Presets
                self.statusBar().showMessage(f"Loaded Preset: {name}")
                
        elif data == "descriptor":
            self.setup_descriptor_ui()
            # Unlocked for custom
            
        else:
            # Legacy
            self.current_spec = None
            self.setup_legacy_ui()

    def _clear_params(self):
        while self.form_param.rowCount() > 0:
            self.form_param.removeRow(0)
        self.param_widgets = {}

    def setup_legacy_ui(self):
        self._clear_params()
        # Alpha
        spin_alpha = self._make_spin(0.30, 0, 1, 0.05)
        self.form_param.addRow("Import Share (α):", spin_alpha)
        self.param_widgets['alpha'] = spin_alpha
        
        # Chi
        spin_chi = self._make_spin(1.0, 0, 5, 0.1)
        self.form_param.addRow("Trade Elast. (χ):", spin_chi)
        self.param_widgets['chi'] = spin_chi

    def setup_descriptor_ui(self):
        self._clear_params()
        if not self.current_spec: return
        
        for pname, pspec in self.current_spec.parameters.items():
            val = pspec.value
            # Heuristic ranges
            vmin = val * 0.1 if val > 0 else val * 2.0
            vmax = val * 2.0 if val > 0 else val * 0.1
            if val == 0: vmin, vmax = -1.0, 1.0
            if pname in ['beta', 'sigma', 'phi_pi']: vmin = 0.0
            
            spin = self._make_spin(val, -1000, 1000, 0.01)
            self.form_param.addRow(f"{pname}:", spin)
            self.param_widgets[pname] = spin

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
        
        # Determine Mode
        # If we have a loaded spec (Custom or Preset), use Descriptor Mode
        is_descriptor = (self.current_spec is not None)
        
        try:
            # Common Experiment Config
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

            if is_descriptor:
                # --- Descriptor Path (Custom OR Preset) ---
                print(f"Running in Descriptor Mode: {self.current_spec.name}")
                
                # Update Spec Params from GUI (Only if widgets exist/enabled)
                # If locked (Preset), this effectively does nothing or re-sets same values.
                for pname, widget in self.param_widgets.items():
                    if pname in self.current_spec.parameters:
                        self.current_spec.parameters[pname].value = widget.value()
                
                # Instantiate Solver
                # Check for Blocks -> SSJ
                if self.current_spec.blocks:
                     print("[GUI] Detected Blocks. Using LinearSSJSolver (HANK Bridge).")
                     solver = LinearSSJSolver(self.current_spec, T=50)
                else:
                     solver = DSGEStaticSolver(self.current_spec, T=50)
                
                # Pass solver as model
                model = solver
                
            else:
                # --- Legacy Path ---
                model_type = self.combo_model.currentData()
                
                # Retrieve params from widgets (safely)
                alpha = self.param_widgets['alpha'].value() if 'alpha' in self.param_widgets else 0.33
                chi = self.param_widgets['chi'].value() if 'chi' in self.param_widgets else 1.0
                
                is_sticky = self.check_sticky.isChecked()
                kappa = 0.1 if is_sticky else 100.0
                params = {'alpha': alpha, 'chi': chi, 'kappa': kappa, 'beta': 0.99, 'phi_pi': 1.5}
                
                model = Model(model_type=model_type, T=50, params=params, cache_dir=self.cache_dir)
            
            # Threading
            self.worker = EngineWorker(model, exp_config)
            self.worker.progress.connect(lambda s: self.lbl_status.setText(s))
            self.worker.error.connect(self.on_error)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
            
        except Exception as e:
            self.on_error(str(e))

    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Error")
        self.lbl_status.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Error", msg)

    def clear_plots(self):
        self.history = []
        self.canvas_agg.fig.clear()
        self.canvas_agg.draw()
        self.canvas_ineq.fig.clear() # Optional: Clear others too
        self.canvas_ineq.draw()
        self.canvas_mech.fig.clear()
        self.canvas_mech.draw()

    def export_plots(self):
        if not self.history:
             QMessageBox.warning(self, "No Data", "Run simulation first.")
             return
             
        fname, _ = QFileDialog.getSaveFileName(self, "Export Figure", "monad_experiment.png", "PNG Image (*.png);;PDF Document (*.pdf)")
        if not fname: return
        
        try:
            from datetime import datetime
            
            # Generate Caption
            labels = [h['label'] for h in self.history]
            scenarios = ", ".join(labels)
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            caption = f"Generated by Monad Studio | {date_str}\nScenarios: {scenarios}"
            
            fig = self.canvas_agg.fig
            
            # Add footer
            fig.subplots_adjust(bottom=0.15)
            text_obj = fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=8, color='#555', family='monospace')
            
            fig.savefig(fname, dpi=150) # Standard save
            
            # Cleanup
            text_obj.remove()
            fig.subplots_adjust(bottom=0.05) # Reset
            self.canvas_agg.draw()
            
            self.statusBar().showMessage(f"Exported to {fname}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def on_finished(self, results):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Complete")
        self.lbl_status.setStyleSheet("color: green;")
        
        # Add to history
        if self.current_spec:
            name = self.current_spec.name
        else:
            name = self.combo_model.currentText()
            
        # Add timestamp or iterator?
        count = len(self.history) + 1
        label = f"#{count} {name}"
        self.history.append({'label': label, 'res': results})
        
        # Plotting
        self.plot_aggregate()
        self.plot_inequality(results) # Inequality usually just latest run
        self.plot_mechanisms(results) # Mechanisms just latest run

    def plot_aggregate(self):
        try:
            canvas = self.canvas_agg
            canvas.fig.clear()
            
            if not self.history:
                canvas.draw()
                return
                
            # Use variables from the LAST run to determine layout
            latest_res = self.history[-1]['res']
            
            # Filter vars
            all_vars = [k for k in latest_res.keys() if not k.startswith('analysis') and not k.startswith('d')]
            priority = ['y', 'pi', 'i', 'Y', 'C_agg']
            vars = []
            for p in priority:
                if p in all_vars:
                    vars.append(p)
                    all_vars.remove(p)
            vars.extend(all_vars[:4-len(vars)])
            
            n = len(vars)
            if n == 0: return
            
            rows = 2 if n > 2 else 1
            cols = 2 if n > 1 else 1
            
            colors = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#f39c12', '#34495e']
            styles = ['-', '--', '-.', ':']
            
            # Loop Subplots
            for i, var in enumerate(vars):
                ax = canvas.fig.add_subplot(rows, cols, i+1)
                ax.set_title(var)
                ax.grid(True, alpha=0.3)
                ax.axhline(0, c='k', lw=0.5)
                
                # Loop History
                for j, item in enumerate(self.history):
                    res = item['res']
                    label = item['label']
                    
                    if var in res:
                        y = res[var]
                        if not isinstance(y, np.ndarray): y = np.array(y)
                        
                        # Auto-scale check (based on latest run usually, or check all?)
                        # Checking latest is safer for consistent scale across lines?
                        # No, transform check should be generic.
                        # If values are small (<0.2), assume %
                        # But mixing large and small might be weird.
                        # Assuming consistent units across comparisons.
                        
                        is_small = (np.max(np.abs(y)) < 0.2 and np.max(np.abs(y)) > 1e-6)
                        if is_small: y = y * 100
                        
                        c = colors[j % len(colors)]
                        ls = styles[(j // len(colors)) % len(styles)]
                        
                        ax.plot(y, label=label, color=c, ls=ls, lw=2, alpha=0.9)
                
                if i == 0: # Legend only on first plot to save space
                    ax.legend(fontsize='small')
                    
            canvas.fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            print(f"Plotting Error: {e}")
            import traceback
            traceback.print_exc()

    def plot_inequality(self, res):
        try:
            canvas = self.canvas_ineq
            canvas.fig.clear()
            
            if 'analysis_ineq' not in res:
                canvas.fig.text(0.5, 0.5, "No Inequality Data", ha='center')
                canvas.draw()
                return
                
            ineq = res['analysis_ineq']
            # Fallback for empty keys
            if 'top10' not in ineq: return
            
            t = np.arange(len(ineq['top10']))
            
            # Left: Group Response
            ax1 = canvas.fig.add_subplot(121)
            ax1.plot(t, ineq['top10']*100, label='Top 10%', color='#e74c3c')
            ax1.plot(t, ineq['bottom50']*100, label='Bottom 50%', color='#3498db')
            if 'debtors' in ineq:
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
        except Exception as e:
            print(f"Inequality Plot Error: {e}")

    def plot_mechanisms(self, res):
        try:
            canvas = self.canvas_mech
            canvas.fig.clear()
            
            if 'analysis_decomp' not in res:
                canvas.fig.text(0.5, 0.5, "No Decomposition Data", ha='center')
                canvas.draw()
                return
                
            decomp = res['analysis_decomp']
            if 'direct' not in decomp: return
            
            top = decomp['direct']
            bottom = decomp['indirect']
            t = np.arange(len(top))
            
            ax = canvas.fig.add_subplot(111)
            ax.bar(t, top, label='Direct Effect (Partial Eq)', color='#f1c40f', alpha=0.8)
            ax.bar(t, bottom, bottom=top, label='Indirect Effect (GE Multiplier)', color='#e67e22', alpha=0.8)
            
            total = top + bottom
            ax.plot(t, total, 'k--', label='Total Effect', lw=2)
            
            ax.set_title("Multiplier Decomposition (PE vs GE)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, c='k', lw=0.5)
            
            canvas.fig.tight_layout()
            canvas.draw()
        except Exception as e:
            print(f"Mechanism Plot Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Optional: Modern Style
    app.setStyle("Fusion")
    
    window = MonadCockpit()
    window.show()
    sys.exit(app.exec())
