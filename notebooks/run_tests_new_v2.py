# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib pytest pytest-cov pandas numpy scikit-learn doubleml catboost joblib statsmodels econml

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys
import os

# Add the src directory to sys.path so insurance_causal is importable
sys.path.insert(0, "/Workspace/insurance-causal/src")

# Verify import works
import insurance_causal
print(f"insurance_causal loaded from: {insurance_causal.__file__}")
print(f"Version: {insurance_causal.__version__}")

# COMMAND ----------
# Run the new test files
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-causal/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

new_test_files = [
    "/Workspace/insurance-causal/tests/test_diagnostics_extended.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_estimator.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_data.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_diagnostics.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_inference.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_targeting.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_exposure.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_clustering.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_estimator_helpers.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_inference_helpers.py",
    "/Workspace/insurance-causal/tests/causal_forest/test_targeting_helpers.py",
    "/Workspace/insurance-causal/tests/elasticity/test_fit.py",
    "/Workspace/insurance-causal/tests/elasticity/test_demand.py",
    "/Workspace/insurance-causal/tests/elasticity/test_diagnostics.py",
    "/Workspace/insurance-causal/tests/elasticity/test_optimise.py",
    "/Workspace/insurance-causal/tests/elasticity/test_surface.py",
    "/Workspace/insurance-causal/tests/elasticity/test_demand_helpers.py",
    "/Workspace/insurance-causal/tests/elasticity/test_fit_helpers.py",
    "/Workspace/insurance-causal/tests/elasticity/test_surface_helpers.py",
]

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        *new_test_files,
        "-v", "--tb=long",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        f"--rootdir=/Workspace/insurance-causal",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-8000:] if len(output) > 8000 else output)
