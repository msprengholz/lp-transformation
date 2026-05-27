#!/usr/bin/env python3
"""Run GPU Viquerat discovery on Colab."""
import subprocess, sys, os

# Install deps silently
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "slangpy", "scipy"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Now run the actual script
os.execv(sys.executable, [sys.executable, "src/gpu_viquerat_discovery.py"])