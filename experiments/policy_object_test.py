import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.model import MonadModel
from monad.policy import Rules

def main():
    print("=== Policy-as-Object Verification ===")
    
    # 0. Mock Run to avoid C++ dependency in unit test
    def mock_run(self, params=None, config_path=None):
        # We just want to inspect overrides
        return {'status': 'mock_ok', 'overrides': self.overrides}
    MonadModel.run = mock_run

    # 1. Initialize Model
    m = MonadModel("dummy_path")
    
    # 2. Attach Policy Object (Inertial Taylor)
    # Using the new Phase 2 API
    policy = Rules.InertialTaylor(phi_pi=2.5, rho=0.9, name="fed_board")
    
    print("\n[Action] Attaching Policy: InertialTaylor(phi=2.5, rho=0.9)")
    m.attach(policy)
    
    # 3. Verify Mutation via Object
    print("\n[Action] Mutating Policy...")
    policy.mutate(phi_pi=3.0)
    
    # 4. Verify Model State
    res = m.run()
    
    print("\n[Audit] Model Overrides:")
    print(res['overrides'])
    
    # Assertions
    assert res['overrides']['rho'] == 0.9, "Initial policy param failed"
    assert res['overrides']['phi_pi'] == 3.0, "Mutation failed"
    assert m.policy_block.name == "fed_board", "Policy attachment failed"
    
    # History Check
    print("\n[Audit] History Log:")
    hist = m.history()
    assert len(hist) > 0
    assert hist[0]['changes']['attached'] == "fed_board"
    
    print("\n=> Policy Object Verification MATCHED expectations.")

if __name__ == "__main__":
    main()
