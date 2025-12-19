from monad.cpp_backend import CppBackend

try:
    print("Initializing CppBackend with model_type='one_asset'...")
    backend = CppBackend(model_type='one_asset')
    
    print("Solving steady state...")
    res = backend.solve_steady_state()
    
    print("OneAsset Solved Successfully.")
    z_g = backend._income['z_grid']
    print(f"Input Z grid type: {type(z_g)}")
    print(f"Input Z grid len: {len(z_g)}")
    print(f"Policy grid size: {len(res['c_pol'])}")
    print(f"Distribution size: {len(res['distribution'])}")
    print(f"Aggregate Liquid Assets: {res['agg_liquid']}")
    # print(res['c_pol'][:10])
    import numpy as np
    print(f"Result c_pol type: {type(res['c_pol'])}")
    if isinstance(res['c_pol'], np.ndarray):
        print(f"Result c_pol shape: {res['c_pol'].shape}")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
