"""
Run insurance-causal tests on Databricks.
Copies source and tests to local disk first to avoid workspace filesystem limitations.

Strategy: use a subprocess to install all dependencies and run pytest in a
fresh Python process. This avoids issues with cached/pre-loaded modules in the
Databricks notebook kernel (e.g. old numpy/scipy that conflict with newer versions).
"""
import os
import sys
import subprocess
import shutil

# ── Step 1: Install dependencies via subprocess (fresh Python process) ───────

install_script = """
import subprocess, sys

def pip(*pkgs, upgrade=False):
    flags = ['--quiet']
    if upgrade:
        flags.append('--upgrade')
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install'] + flags + list(pkgs),
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print('pip stderr:', r.stderr[-500:])
    return r.returncode == 0

# Install numpy first (needs >=1.25 for numpy.exceptions)
pip('numpy>=1.25', upgrade=True)
pip('scipy>=1.11', upgrade=True)
pip(
    'doubleml', 'catboost', 'polars', 'pyarrow', 'scikit-learn>=1.3',
    'joblib', 'matplotlib', 'jinja2', 'pytest', 'pytest-cov',
    upgrade=False
)

econml_ok = pip('econml')
print('econml_ok:', econml_ok)

import numpy as np
import scipy
print(f'numpy {np.__version__}, scipy {scipy.__version__}')
"""

r = subprocess.run([sys.executable, "-c", install_script], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("Install stderr:", r.stderr[-2000:])
    raise RuntimeError("Dependency installation failed")

# Check if econml installed
econml_ok = "econml_ok: True" in r.stdout

# ── Step 2: Copy source and tests to local disk ───────────────────────────────

local_root = "/tmp/insurance-causal"
if os.path.exists(local_root):
    shutil.rmtree(local_root)

shutil.copytree("/Workspace/insurance-causal/src", f"{local_root}/src")
shutil.copytree("/Workspace/insurance-causal/tests", f"{local_root}/tests")
print(f"Copied to {local_root}")

src_path = f"{local_root}/src"

# ── Step 3: Run pytest in subprocess ─────────────────────────────────────────

test_args = [
    sys.executable, "-m", "pytest",
    f"{local_root}/tests/",
    "-v", "--tb=long",  # long tracebacks so we can see errors
]
if not econml_ok:
    print("WARNING: econml unavailable, skipping test_fit and test_optimise")
    test_args += [
        f"--ignore={local_root}/tests/elasticity/test_fit.py",
        f"--ignore={local_root}/tests/elasticity/test_optimise.py",
    ]

env = os.environ.copy()
env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")

result = subprocess.run(test_args, capture_output=True, text=True, env=env)
output = result.stdout + result.stderr
print("=== TEST RESULTS ===")
print(output[-15000:] if len(output) > 15000 else output)
print(f"Return code: {result.returncode}")

lines = output.strip().split("\n")
summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "no summary")
status = "PASSED" if result.returncode == 0 else "FAILED"

if result.returncode != 0:
    # Raise exception so the error trace is visible in the Databricks run output
    raise RuntimeError(f"Tests FAILED: {summary}\n\nLast 2000 chars:\n{output[-2000:]}")

try:
    dbutils.notebook.exit(f"PASSED: {summary}")
except NameError:
    pass
