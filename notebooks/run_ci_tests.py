# Databricks notebook source
# MAGIC %pip install insurance-causal[causal_forest] pytest --quiet

# COMMAND ----------

# Install the local patched version from PyPI is not yet available.
# Instead, install directly from source uploaded to DBFS.
# We use pip install with the source directory.

import subprocess, sys

# The source was uploaded to /Workspace via import-dir.
# Install in editable mode from the workspace path.
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/Users/pricing.frontier@gmail.com/insurance-causal-ci",
     "--quiet", "--no-deps"],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if result.stdout else "")
print(result.stderr[-3000:] if result.stderr else "")

# COMMAND ----------

# Also install dependencies
result2 = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "econml>=0.15", "catboost>=1.2", "pytest>=8.0",
     "--quiet"],
    capture_output=True, text=True
)
print(result2.stdout[-3000:] if result2.stdout else "")
print(result2.stderr[-3000:] if result2.stderr else "")

# COMMAND ----------

import subprocess, sys, os

result3 = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/Users/pricing.frontier@gmail.com/insurance-causal-ci/tests/causal_forest/",
     "-x", "-q", "--tb=short"],
    capture_output=True, text=True,
    cwd="/Workspace/Users/pricing.frontier@gmail.com/insurance-causal-ci"
)
print("STDOUT:")
print(result3.stdout[-5000:])
print("STDERR:")
print(result3.stderr[-2000:])
print("Return code:", result3.returncode)
