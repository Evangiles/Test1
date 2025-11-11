"""Debug script for TabPFN import issues."""
import sys
import os

print("=== TabPFN Import Debug ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print("\nPython path:")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")

print("\n--- Attempting TabPFN import ---")
try:
    # First try importing dependencies
    print("Checking torch...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch not found: {e}")
        
    print("Checking numpy...")
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy not found: {e}")
        
    print("Checking sklearn...")
    try:
        import sklearn
        print(f"  ✓ sklearn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ sklearn not found: {e}")

    print("\nChecking tabpfn_common_utils...")
    try:
        import tabpfn_common_utils
        print("  ✓ tabpfn_common_utils found")
    except ImportError as e:
        print(f"  ✗ tabpfn_common_utils not found: {e}")
        
    print("\nAttempting main import...")
    from tabpfn import TabPFNRegressor
    print("✓ SUCCESS: TabPFNRegressor imported successfully!")
    
    # Try creating instance
    print("\nTrying to create TabPFNRegressor instance...")
    model = TabPFNRegressor(n_estimators=1, device="cpu")
    print("✓ TabPFNRegressor instance created successfully!")
    
except ImportError as e:
    print(f"✗ IMPORT ERROR: {e}")
    print(f"\nFull error details: {repr(e)}")
except Exception as e:
    print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n=== End Debug ===")
