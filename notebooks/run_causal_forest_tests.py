# Databricks notebook source
# COMMAND ----------
import subprocess, sys, os, shutil

# Install dependencies
pkgs = ["polars", "pyarrow", "catboost", "econml==0.15.1", "doubleml", "matplotlib", "pytest"]
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
TMP_DIR = "/tmp/ic_tests"
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR)

# Ignore __pycache__ directories — they can't be read from the Workspace filesystem
ignore_pycache = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")

shutil.copytree("/Workspace/insurance-causal/src", f"{TMP_DIR}/src", ignore=ignore_pycache)
shutil.copytree("/Workspace/insurance-causal/tests", f"{TMP_DIR}/tests", ignore=ignore_pycache)

# Write a minimal setup.cfg (NOT pytest.ini or pyproject.toml) to make /tmp/ic_tests the rootdir.
# Crucially: do NOT copy pyproject.toml, which would cause pytest to use workspace paths.
with open(f"{TMP_DIR}/setup.cfg", "w") as f:
    f.write("[tool:pytest]\ntestpaths = tests\naddopts = --tb=short\n")

# Write a top-level conftest.py that anchors rootdir at TMP_DIR
with open(f"{TMP_DIR}/conftest.py", "w") as f:
    f.write("# Root conftest — anchors pytest rootdir at /tmp/ic_tests\n")

SRC_PATH = f"{TMP_DIR}/src"
TESTS_PATH = f"{TMP_DIR}/tests"

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
import insurance_causal
print("insurance_causal:", insurance_causal.__version__)

# COMMAND ----------
LOG_FILE = "/Workspace/insurance-causal/test_cf_full_output.txt"
env = {
    **os.environ,
    "PYTHONPATH": SRC_PATH,
    "PYTHONDONTWRITEBYTECODE": "1",
}

# Run pytest from TMP_DIR — rootdir will be /tmp/ic_tests, conftest loaded from there
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{TESTS_PATH}/causal_forest/test_data.py",
     f"{TESTS_PATH}/causal_forest/test_exposure.py",
     f"{TESTS_PATH}/causal_forest/test_estimator.py",
     f"{TESTS_PATH}/causal_forest/test_inference.py",
     f"{TESTS_PATH}/causal_forest/test_targeting.py",
     f"{TESTS_PATH}/causal_forest/test_diagnostics.py",
     f"--rootdir={TMP_DIR}",
     f"--confcutdir={TMP_DIR}",
     "--import-mode=importlib",
     "--override-ini=testpaths=",
     "-v", "--tb=short",
     "-p", "no:cacheprovider",
    ],
    capture_output=True, text=True, env=env, timeout=900, cwd=TMP_DIR,
)
cf_rc = result.returncode

with open(LOG_FILE, "w") as f:
    f.write("=== CAUSAL FOREST TESTS ===\n")
    f.write(result.stdout + "\n" + result.stderr)
    f.write(f"\nRETURN CODE: {cf_rc}\n")

print(f"CF tests rc={cf_rc}")
print(result.stdout[-8000:])
if result.stderr.strip():
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------
# Existing tests — also from /tmp
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{TESTS_PATH}/test_treatments.py",
     f"{TESTS_PATH}/test_utils.py",
     f"{TESTS_PATH}/test_diagnostics.py",
     f"{TESTS_PATH}/test_p0_regressions.py",
     f"--rootdir={TMP_DIR}",
     f"--confcutdir={TMP_DIR}",
     "--import-mode=importlib",
     "--override-ini=testpaths=",
     "-q", "--tb=line",
     "-p", "no:cacheprovider",
    ],
    capture_output=True, text=True, env=env, timeout=300, cwd=TMP_DIR,
)
existing_rc = result.returncode

with open(LOG_FILE, "a") as f:
    f.write("\n\n=== EXISTING BASE TESTS ===\n")
    f.write(result.stdout + "\n" + result.stderr)
    f.write(f"\nRETURN CODE: {existing_rc}\n")

print(f"Existing rc={existing_rc}")
print(result.stdout[-2000:])

# COMMAND ----------
summary = f"\nCF: {'PASS' if cf_rc == 0 else 'FAIL'}\nEXISTING: {'PASS' if existing_rc == 0 else 'FAIL'}"
print(summary)
if cf_rc != 0 or existing_rc != 0:
    raise RuntimeError(summary)
print("ALL PASS")
