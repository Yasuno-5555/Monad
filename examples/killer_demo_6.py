"""
Killer Demo 6: Unified Engine Verification
Verifies that the new Config-Driven Architecture correctly adapts to dimensionality.
"""
import os
import json
from monad import Monad

def test_unified_engine():
    print("=== Killer Demo 6: Unified Engine Verification ===")
    
    # 1. Initialize Standard Model
    m = Monad("unified_test", binary_path="bin/MonadEngine")
    
    # 2. Add Assets (The New API)
    print("Adding Asset: Crypto...")
    m.add_asset("crypto")
    
    # 3. Add Beliefs
    print("Adding State: Belief...")
    m.add_state("belief")
    
    # Check Internal State
    print(f"Model Assets: {m.assets}")
    print(f"Model States: {m.state_vars}")
    
    assert "crypto" in m.assets
    assert "belief" in m.state_vars
    
    # Pre-clean stale config
    config_path = os.path.join(m.working_dir, "test_model.json")
    if os.path.exists(config_path):
        os.remove(config_path)

    # 4. Dry Run (Generate Config)
    # We mock subprocess.run to avoid actual execution loop, 
    # but we want to verify config generation.
    # Actually, let's run it. If binary is missing, it will fail gracefully (Safe Mode).
    # We just want to check config.json content.
    
    print("\n[Action] Running Engine (Dry Run)...")
    try:
        m.run(params={"test_run": True})
    except Exception as e:
        print(f"Run ignored (expected since binary might not be built): {e}")

    # 5. Verify Config Output
    config_path = os.path.join(m.working_dir, "test_model.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        params = data.get('parameters', {})
        n_assets = params.get('n_assets')
        has_belief = params.get('has_belief')
        crypto_flag = params.get('crypto')
        
        print("\n[Verification] Config Parameters:")
        print(f"  n_assets:   {n_assets} (Expected: 3)")
        print(f"  has_belief: {has_belief} (Expected: True)")
        print(f"  crypto:     {crypto_flag} (Expected: True)")
        
        if n_assets == 3 and has_belief and crypto_flag:
            print("\n✅ PASS: Config correctly reflects Unified Model state.")
        else:
            print("\n❌ FAIL: Config mismatch.")
            exit(1)
    else:
        print("\n❌ FAIL: Config file not found.")
        exit(1)

if __name__ == "__main__":
    test_unified_engine()
