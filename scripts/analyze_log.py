
import os
import sys

def analyze():
    log_file = "build/demo_fail.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    encodings = ['utf-8', 'cp932', 'latin1', 'utf-16']
    content = ""
    
    for enc in encodings:
        try:
            with open(log_file, encoding=enc, errors='replace') as f:
                content = f.read()
            break
        except Exception:
            continue
            
    lines = content.splitlines()
    errors = [line for line in lines if "error" in line.lower() or "fatal" in line.lower()]
    
    # Print last 20 errors
    print("\n".join(errors[-20:]))

    # Also print any lines related to main_two_asset.cpp near the end
    relevant = [line for line in lines if "main_two_asset" in line or "IncomeProcessFactory" in line]
    print("\n--- Relevant Files Context ---")
    print("\n".join(relevant[-20:]))

if __name__ == "__main__":
    analyze()
