
import os

encodings = ['cp932', 'shift_jis', 'utf-8', 'latin-1']
log_file = "build/build_error.log"

print(f"Reading {log_file}...")

for enc in encodings:
    try:
        print(f"--- Trying encoding: {enc} ---")
        with open(log_file, "r", encoding=enc) as f:
            lines = f.readlines()
            found = False
            for i, line in enumerate(lines):
                if "error" in line.lower() or "エラー" in line:
                    print(f"[{i+1}] {line.strip()}")
                    # Print context
                    for j in range(max(0, i-2), min(len(lines), i+3)):
                        if i != j:
                            print(f"    {lines[j].strip()}")
                    found = True
            if found:
                print("--- Found with this encoding ---")
                break
    except Exception as e:
        print(f"Failed with {enc}: {e}")
