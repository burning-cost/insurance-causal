"""
Run insurance-causal tests on Databricks.
Copies source and tests to local disk first to avoid workspace filesystem limitations.
"""
import os
import sys
import subprocess
import shutil

def pip_install(*pkgs, check=True):
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + list(pkgs),
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("pip stderr:", r.stderr[-500:])
        if check:
            raise RuntimeError(f"pip install failed: {pkgs}")
    return r.returncode == 0

pip_install(
    "doubleml", "catboost", "polars", "pyarrow", "scikit-learn",
    "scipy", "numpy", "joblib", "matplotlib", "jinja2",
    "pytest", "pytest-cov"
)
print("Core deps installed.")

econml_ok = pip_install("econml", check=False)
if not econml_ok:
    print("WARNING: econml unavailable, skipping test_fit and test_optimise")

# Copy to local disk (workspace NFS has restrictions with pytest)
local_root = "/tmp/insurance-causal"
if os.path.exists(local_root):
    shutil.rmtree(local_root)

shutil.copytree("/Workspace/insurance-causal/src", f"{local_root}/src")
shutil.copytree("/Workspace/insurance-causal/tests", f"{local_root}/tests")
print(f"Copied to {local_root}")

src_path = f"{local_root}/src"
sys.path.insert(0, src_path)

import insurance_causal
print(f"insurance_causal {insurance_causal.__version__} loaded")

from insurance_causal.autodml import PremiumElasticity, ForestRiesz, SyntheticContinuousDGP
print("AutoDML imports OK")
from insurance_causal.elasticity import ElasticitySurface, make_renewal_data
print("Elasticity imports OK")

# Build pytest args
test_args = [
    sys.executable, "-m", "pytest",
    f"{local_root}/tests/",
    "-v", "--tb=short",
]
if not econml_ok:
    test_args += [
        f"--ignore={local_root}/tests/elasticity/test_fit.py",
        f"--ignore={local_root}/tests/elasticity/test_optimise.py",
    ]

env = os.environ.copy()
env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")

result = subprocess.run(test_args, capture_output=True, text=True, env=env)
output = result.stdout + result.stderr
print("=== TEST RESULTS ===")
print(output[-10000:] if len(output) > 10000 else output)
print(f"Return code: {result.returncode}")

lines = output.strip().split("\n")
summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "no summary")
status = "PASSED" if result.returncode == 0 else "FAILED"
exit_msg = f"{status}: {summary}"

try:
    dbutils.notebook.exit(exit_msg)
except NameError:
    sys.exit(result.returncode)
