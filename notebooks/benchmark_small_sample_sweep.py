# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: Small-Sample Sweep — DML v0.3.0 Adaptive vs v0.2.x Fixed Params
# MAGIC
# MAGIC Measures absolute bias at 1k-50k policies for:
# MAGIC - Naive Poisson GLM
# MAGIC - DML with adaptive params (v0.3.0 default)
# MAGIC - DML with fixed params (v0.2.x behaviour via nuisance_params override)
# MAGIC
# MAGIC True treatment effect: -0.15. DGP: telematics confounding.

# COMMAND ----------

import subprocess, sys, os, shutil

install_script = """
import subprocess, sys
def pip(*pkgs, upgrade=False):
    flags = ['--quiet']
    if upgrade:
        flags.append('--upgrade')
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + flags + list(pkgs), capture_output=True)
pip('numpy>=1.25', upgrade=True)
pip('scipy>=1.11', upgrade=True)
pip('statsmodels', 'catboost', 'pandas', 'doubleml', 'scikit-learn>=1.3',
    'polars', 'pyarrow', 'joblib')
import numpy as np, scipy
print(f'numpy {np.__version__}, scipy {scipy.__version__}')
"""
r = subprocess.run([sys.executable, "-c", install_script], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-500:])

# COMMAND ----------

local_root = "/tmp/insurance-causal-bench"
if os.path.exists(local_root):
    shutil.rmtree(local_root)
shutil.copytree("/Workspace/insurance-causal/src", f"{local_root}/src")
print(f"Source copied to {local_root}")

# Write the sweep script to a file (avoids f-string escaping issues)
sweep_code = r"""
import sys, warnings, time
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import importlib.util, types

LOCAL_ROOT = sys.argv[1]

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = f'{LOCAL_ROOT}/src/insurance_causal'
pkg = types.ModuleType('insurance_causal')
pkg.__path__ = [base]
pkg.__package__ = 'insurance_causal'
pkg.__spec__ = None
sys.modules['insurance_causal'] = pkg

_utils     = load_mod('insurance_causal._utils',    f'{base}/_utils.py')
treatments = load_mod('insurance_causal.treatments', f'{base}/treatments.py')
_model     = load_mod('insurance_causal._model',    f'{base}/_model.py')
pkg._utils = _utils
pkg.treatments = treatments

CausalPricingModel       = _model.CausalPricingModel
BinaryTreatment          = treatments.BinaryTreatment
adaptive_catboost_params = _utils.adaptive_catboost_params

TRUE_EFFECT = -0.15
BASE_FREQ   = 0.12
RNG_SWEEP   = np.random.default_rng(2024)

def make_dgp(n, rng):
    age     = rng.uniform(21, 75, n)
    val_log = rng.normal(10.2, 0.7, n)
    pc_risk = rng.beta(2, 3, n)
    exp_yrs = rng.uniform(0.5, 1.0, n)

    age_s   = (age - age.mean()) / age.std()
    val_s   = (val_log - val_log.mean()) / val_log.std()
    risk_s  = (pc_risk - pc_risk.mean()) / pc_risk.std()
    safety  = 0.4 * age_s - 0.3 * val_s - 0.5 * risk_s
    prop    = 1 / (1 + np.exp(-(0.8 * safety - 0.3)))
    treat   = rng.binomial(1, prop).astype(float)

    log_mu  = (np.log(BASE_FREQ) - 0.3*age_s + 0.2*val_s + 0.4*risk_s
               + TRUE_EFFECT*treat + np.log(exp_yrs))
    claims  = rng.poisson(np.exp(log_mu))

    return pd.DataFrame({
        'claims': claims, 'exposure': exp_yrs, 'treat': treat,
        'age': age, 'val_log': val_log, 'pc_risk': pc_risk,
    })

SAMPLE_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
CONFOUNDERS  = ['age', 'val_log', 'pc_risk']
TREATMENT    = BinaryTreatment(column='treat')
FIXED_PARAMS = {'iterations': 500, 'depth': 6, 'learning_rate': 0.05}

rows = []
for n in SAMPLE_SIZES:
    print(f"\nn={n:,}...")
    df = make_dgp(n, RNG_SWEEP)

    glm = smf.glm('claims ~ treat + age + val_log + pc_risk', data=df,
                  family=smf.families.Poisson(), exposure=df['exposure']).fit(disp=False)
    naive_est  = float(glm.params['treat'])
    naive_bias = abs(naive_est - TRUE_EFFECT)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_a = CausalPricingModel(
            outcome='claims', outcome_type='poisson',
            treatment=TREATMENT, confounders=CONFOUNDERS,
            exposure_col='exposure', cv_folds=5, random_state=42,
        )
        m_a.fit(df)
    t_adapt = time.time() - t0
    adapt_est  = m_a.average_treatment_effect().estimate
    adapt_bias = abs(adapt_est - TRUE_EFFECT)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_f = CausalPricingModel(
            outcome='claims', outcome_type='poisson',
            treatment=TREATMENT, confounders=CONFOUNDERS,
            exposure_col='exposure', cv_folds=5, random_state=42,
            nuisance_params=FIXED_PARAMS,
        )
        m_f.fit(df)
    t_fixed = time.time() - t0
    fixed_est  = m_f.average_treatment_effect().estimate
    fixed_bias = abs(fixed_est - TRUE_EFFECT)

    chosen = adaptive_catboost_params(n)
    print(f"  naive_bias={naive_bias:.4f} | adapt_bias={adapt_bias:.4f} | fixed_bias={fixed_bias:.4f}")
    print(f"  adaptive: iter={chosen['iterations']}, depth={chosen['depth']}, t={t_adapt:.1f}s | fixed t={t_fixed:.1f}s")

    rows.append({
        'n': n,
        'naive_est': round(naive_est, 4),
        'naive_bias': round(naive_bias, 4),
        'dml_adaptive': round(adapt_est, 4),
        'dml_adaptive_bias': round(adapt_bias, 4),
        'dml_fixed': round(fixed_est, 4),
        'dml_fixed_bias': round(fixed_bias, 4),
        'adaptive_iterations': chosen['iterations'],
        'adaptive_depth': chosen['depth'],
        'adaptive_time_s': round(t_adapt, 1),
        'fixed_time_s': round(t_fixed, 1),
    })

sweep_df = pd.DataFrame(rows)
print("\n=== FINAL SWEEP RESULTS ===")
print(f"True effect: {TRUE_EFFECT}")
print(sweep_df.to_string(index=False))

small_rows = sweep_df[sweep_df['n'] <= 10_000]
adaptive_wins = (small_rows['dml_adaptive_bias'] < small_rows['dml_fixed_bias']).sum()
total_small = len(small_rows)
print(f"\nAdaptive params win at {adaptive_wins}/{total_small} small-n settings")
print("SWEEP COMPLETE")
"""

sweep_path = f"{local_root}/sweep_script.py"
with open(sweep_path, "w") as f:
    f.write(sweep_code)
print(f"Sweep script written to {sweep_path}")

# COMMAND ----------

env = os.environ.copy()
env["PYTHONPATH"] = f"{local_root}/src:" + env.get("PYTHONPATH", "")

r2 = subprocess.run(
    [sys.executable, sweep_path, local_root],
    capture_output=True, text=True, env=env, timeout=900
)
print("STDOUT:")
out = r2.stdout
print(out[-8000:] if len(out) > 8000 else out)
if r2.returncode != 0:
    print("STDERR:", r2.stderr[-3000:])
    raise RuntimeError("Sweep FAILED")

# COMMAND ----------

try:
    dbutils.notebook.exit("PASSED: small-sample sweep complete")
except NameError:
    pass
