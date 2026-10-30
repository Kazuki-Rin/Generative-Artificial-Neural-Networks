#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FORGE NEAT GENOMES - Thin Wrapper (Deprecated Entry Point)
==========================================================
This script is kept for backwards compatibility (CI/CD scripts).
It delegates to the canonical forge_genesis.py which handles ALL models
(NEAT, PyTorch, M0 JSON, M8 ART) with a single deterministic seed.
"""

import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def main():
    print("=" * 70)
    print("⚠️  FORGE_NEAT_GENOMES.PY IS DEPRECATED")
    print("   Redirecting to canonical forge_genesis.py (handles ALL models)...")
    print("=" * 70)
    
    # Import and run the canonical entry point
    try:
        # Ensure we are in the right directory for relative paths in forge_genesis
        os.chdir(BASE_DIR)
        
        from forge_genesis import forge_real_neat
        forge_real_neat()
        
    except ImportError as e:
        print(f"💀 FATAL: Cannot import forge_genesis: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💀 FATAL: Forge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()