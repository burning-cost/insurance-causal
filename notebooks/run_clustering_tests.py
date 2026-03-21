# Databricks notebook source
# COMMAND ----------
import subprocess, sys, os, shutil

# Install dependencies (pin econml version for reproducibility)
pkgs = ["polars", "pyarrow", "catboost", "econml==0.15.1", "doubleml", "pytest", "scikit-learn>=1.3", "scipy>=1.11"]
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs,
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDOUT:", result.stdout[-500:])
    print("STDERR:", result.stderr[-500:])
    raise RuntimeError("pip install failed")
print("Dependencies OK")

result = subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True, text=True)
print("pytest version:", result.stdout.strip())

# COMMAND ----------
# Copy everything to /tmp — completely isolated from the workspace filesystem
TMP_DIR = "/tmp/ic_clustering_tests"
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR)

ignore_pycache = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

shutil.copytree("/Workspace/insurance-causal/src", f"{TMP_DIR}/src", ignore=ignore_pycache)
shutil.copytree("/Workspace/insurance-causal/tests", f"{TMP_DIR}/tests", ignore=ignore_pycache)

with open(f"{TMP_DIR}/setup.cfg", "w") as f:
    f.write("[tool:pytest]\ntestpaths = tests\naddopts = --tb=short\n")

with open(f"{TMP_DIR}/conftest.py", "w") as f:
    f.write("# Root conftest — anchors pytest rootdir at /tmp/ic_clustering_tests\n")

SRC_PATH = f"{TMP_DIR}/src"
TESTS_PATH = f"{TMP_DIR}/tests"

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
import insurance_causal
print("insurance_causal:", insurance_causal.__version__)

from insurance_causal.causal_forest.clustering import CausalClusteringAnalyzer
print("CausalClusteringAnalyzer import OK")

# COMMAND ----------
LOG_FILE = "/Workspace/insurance-causal/test_clustering_output.txt"
env = {
    **os.environ,
    "PYTHONPATH": SRC_PATH,
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{TESTS_PATH}/causal_forest/test_clustering.py",
     f"--rootdir={TMP_DIR}",
     f"--confcutdir={TMP_DIR}",
     "--import-mode=importlib",
     "--override-ini=testpaths=",
     "-v", "--tb=long",
     "-p", "no:cacheprovider",
    ],
    capture_output=True, text=True, env=env, timeout=900, cwd=TMP_DIR,
)
rc = result.returncode

with open(LOG_FILE, "w") as f:
    f.write("=== CLUSTERING TESTS ===\n")
    f.write(result.stdout + "\n" + result.stderr)
    f.write(f"\nRETURN CODE: {rc}\n")

print(f"Clustering tests rc={rc}")
print(result.stdout[-12000:])
if result.stderr.strip():
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------
lines = result.stdout.strip().split("\n")
summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "no summary")
print(f"Summary: {summary}")

if rc != 0:
    raise RuntimeError(f"Tests FAILED: {summary}")

try:
    dbutils.notebook.exit(f"PASSED: {summary}")
except NameError:
    print(f"PASSED: {summary}")
