# Databricks notebook source
# COMMAND ----------
import subprocess
import sys

# Install all dependencies
pkgs = [
    "doubleml", "catboost", "polars", "pyarrow", "scikit-learn",
    "scipy", "numpy", "joblib", "econml", "matplotlib", "jinja2",
    "pytest", "pytest-cov"
]
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs,
    capture_output=True, text=True
)
print("pip install stdout:", result.stdout[-2000:] if result.stdout else "(none)")
if result.returncode != 0:
    print("pip install stderr:", result.stderr[-2000:])
    raise RuntimeError("pip install failed")
print("Dependencies installed.")

# COMMAND ----------
# Install the package from the uploaded source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-causal/src", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "(none)")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("package install failed")
print("insurance-causal installed.")

# COMMAND ----------
# Smoke test imports
import insurance_causal
print("insurance_causal version:", insurance_causal.__version__)

from insurance_causal import CausalPricingModel, AverageTreatmentEffect
print("Core imports OK")

from insurance_causal.autodml import (
    PremiumElasticity, DoseResponseCurve, PolicyShiftEffect,
    SelectionCorrectedElasticity, ForestRiesz, LinearRiesz,
    SyntheticContinuousDGP, EstimationResult, OutcomeFamily,
)
print("AutoDML imports OK")

from insurance_causal.elasticity import (
    RenewalElasticityEstimator, ElasticitySurface, RenewalPricingOptimiser,
    ElasticityDiagnostics, demand_curve, make_renewal_data,
)
print("Elasticity imports OK")

# COMMAND ----------
# Run base causal tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/test_treatments.py",
     "/Workspace/insurance-causal/tests/test_utils.py",
     "/Workspace/insurance-causal/tests/test_diagnostics.py",
     "-v", "--tb=short"],
    capture_output=True, text=True
)
print("=== BASE CAUSAL TESTS ===")
print(result.stdout[-5000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

# COMMAND ----------
# Run autodml tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/autodml/",
     "-v", "--tb=short"],
    capture_output=True, text=True
)
print("=== AUTODML TESTS ===")
print(result.stdout[-8000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
autodml_rc = result.returncode

# COMMAND ----------
# Run elasticity tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/elasticity/",
     "-v", "--tb=short"],
    capture_output=True, text=True
)
print("=== ELASTICITY TESTS ===")
print(result.stdout[-8000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
elasticity_rc = result.returncode

# COMMAND ----------
# Final summary
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/",
     "--tb=line", "-q"],
    capture_output=True, text=True
)
print("=== FULL SUITE SUMMARY ===")
print(result.stdout[-5000:])
print("Return code:", result.returncode)

if result.returncode == 0:
    print("\nALL TESTS PASSED")
else:
    print("\nSOME TESTS FAILED - see above")
