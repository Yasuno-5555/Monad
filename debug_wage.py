import os
import shutil
from monad.model import MonadModel

def clean_outputs():
    for f in ["transition_nk.csv", "steady_state.csv"]:
        if os.path.exists(f):
            os.remove(f)

def run_debug(mode, theta):
    print(f"\n[{mode}] Running theta_w={theta}")
    clean_outputs()
    
    m = MonadModel(f"debug_{mode}")
    # Setup minimal model
    m.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
    m.set_unemployment(u_rate=0.05, replacement_rate=0.4)
    m.set_fiscal(tau=0.15)
    m.define_grid(size=200)
    m.set_param("theta_w", theta)
    
    # Run
    # We want to see stdout, so we run directly or trust MonadModel prints?
    # MonadModel.solve() captures stdout/stderr usually?
    try:
        res = m.solve()
        if "transition" in res:
            dr_min = res["transition"]["dr"].min()
            print(f"[{mode}] Success. Min dr: {dr_min}")
        else:
            print(f"[{mode}] No transition data.")
    except Exception as e:
        print(f"[{mode}] Error: {e}")

if __name__ == "__main__":
    run_debug("Sticky", 0.75)
    run_debug("Flex", 0.01)
