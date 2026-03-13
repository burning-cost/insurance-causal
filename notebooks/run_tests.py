# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-causal test suite
# MAGIC Run all tests after the autodml + elasticity merge.

# COMMAND ----------
%pip install doubleml catboost polars pyarrow scikit-learn scipy numpy joblib econml matplotlib jinja2 pytest pytest-cov --quiet

# COMMAND ----------
import subprocess, sys

# Install the package from the uploaded source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-causal/src", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------
# Quick import smoke test
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
# Run the base causal tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/test_treatments.py",
     "/Workspace/insurance-causal/tests/test_utils.py",
     "/Workspace/insurance-causal/tests/test_diagnostics.py",
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True
)
print(result.stdout[-5000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------
# Run autodml tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/autodml/",
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True
)
print(result.stdout[-8000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------
# Run elasticity tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/elasticity/",
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True
)
print(result.stdout[-8000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------
# Full suite summary
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-causal/tests/",
     "--tb=short", "-q"],
    capture_output=True, text=True
)
print(result.stdout[-5000:])
print("Return code:", result.returncode)
