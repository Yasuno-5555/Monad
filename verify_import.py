import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    import monad.monad_core as mc
    print("Import successful!")
    print("Methods:", dir(mc))
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
