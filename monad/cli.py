"""
Monad CLI
Command-line interface for the Monad "Thinking Library".
Design Rule: Must ONLY access Facade API (monad.facade). No backend access.
"""

import argparse
import sys
import os
from monad import Monad

def main():
    parser = argparse.ArgumentParser(description="Monad Studio CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # --- RUN Command ---
    run_parser = subparsers.add_parser("run", help="Run a simulation from a preset")
    run_parser.add_argument("preset", help="Name of the preset (e.g., us_normal, japan_zlb)")
    run_parser.add_argument("--shock", type=str, help="Shock definition (name:size, e.g., monetary:-0.01)")
    run_parser.add_argument("--nonlinear", action="store_true", help="Enable nonlinear solver")
    run_parser.add_argument("--zlb", action="store_true", help="Enable ZLB constraint (implies --nonlinear)")
    run_parser.add_argument("--export", type=str, help="Path to export results (e.g., result.pdf, result.csv)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        try:
            run_simulation(args)
        except Exception as e:
            # User-Agent Error Reporting Standard
            print(f"\n[Error] Simulation failed to complete.")
            print(f"Details: {str(e)}")
            sys.exit(1)
            
    else:
        parser.print_help()

def run_simulation(args):
    """
    Orchestrate the thought process via Facade.
    """
    print(f"--- Monad CLI: Running {args.preset} ---")
    
    # 1. Initialize
    m = Monad(args.preset)
    
    # 2. Setup Shocks
    if args.shock:
        try:
            name, size_str = args.shock.split(":")
            size = float(size_str)
            m.shock(name, size=size)
            print(f"Added Shock: {name} = {size}")
        except ValueError:
            raise ValueError("Invalid shock format. Use name:val (e.g., monetary:-0.01)")
    else:
        # Default shock if none provided? Or just error?
        # Monad needs a shock to think.
        print("No shock specified, applying default monetary shock (-0.01).")
        m.shock("monetary", -0.01)

    # 3. Solve (The Thought)
    # ZLB implies nonlinear
    if args.zlb:
        args.nonlinear = True
        
    res = m.solve(nonlinear=args.nonlinear, zlb=args.zlb)
    
    # 4. Diagnostics
    det = res.determinacy()
    print(f"Status: {det['status']}")
    print(f"Notes:  {det['notes']}")
    
    # 5. Export / Output
    if args.export:
        res.export(args.export)
    else:
        print("\nNote: No --export path provided. Simulation finished but results not saved.")
        print("Use --export result.pdf or --export result.csv to save.")

if __name__ == "__main__":
    main()
