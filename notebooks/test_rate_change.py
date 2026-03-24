# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-causal v0.6.0: RateChangeEvaluator Tests
# MAGIC
# MAGIC Runs the full test suite for the rate_change module.

# COMMAND ----------

# MAGIC %pip install statsmodels>=0.14 matplotlib>=3.7 pytest

# COMMAND ----------

import subprocess
import sys
import os

# Install the package from workspace
os.chdir("/Workspace/insurance-causal-v060")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------

# Run pytest on the rate_change test suite
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-causal-v060/tests/rate_change/",
        "-v", "--tb=short", "--no-header",
        "-p", "no:cacheprovider",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-causal-v060",
)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"Tests failed (exit code {result.returncode})")
else:
    print("\nAll tests passed.")
