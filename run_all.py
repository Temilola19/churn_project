"""Reproduce everything: python run_all.py"""
import subprocess, sys
for s in ["01_clean.py", "02_eda.py", "03_model.py"]:
    print(f"\n=== {s} ===")
    subprocess.run([sys.executable, s], cwd="src", check=True)
