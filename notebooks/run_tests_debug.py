"""
Debug failing tests — capture output to file.
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
    if r.returncode != 0 and check:
        raise RuntimeError(f"pip install failed: {pkgs}\n{r.stderr[-300:]}")
    return r.returncode == 0

pip_install("doubleml", "catboost", "polars", "pyarrow", "scikit-learn",
            "scipy", "numpy", "joblib", "matplotlib", "jinja2", "pytest", "pytest-cov")
econml_ok = pip_install("econml", check=False)

local_root = "/tmp/insurance-causal"
if os.path.exists(local_root):
    shutil.rmtree(local_root)
shutil.copytree("/Workspace/insurance-causal/src", f"{local_root}/src")
shutil.copytree("/Workspace/insurance-causal/tests", f"{local_root}/tests")

src_path = f"{local_root}/src"
sys.path.insert(0, src_path)
env = os.environ.copy()
env["PYTHONPATH"] = src_path + ":" + env.get("PYTHONPATH", "")

test_args = [
    sys.executable, "-m", "pytest",
    f"{local_root}/tests/",
    "--tb=short", "-q",
]
if not econml_ok:
    test_args += [
        f"--ignore={local_root}/tests/elasticity/test_fit.py",
        f"--ignore={local_root}/tests/elasticity/test_optimise.py",
    ]

result = subprocess.run(test_args, capture_output=True, text=True, env=env)
full_output = result.stdout + result.stderr

# Write to workspace file so we can read it back
out_path = "/Workspace/insurance-causal/test_output.txt"
with open(out_path, "w") as f:
    f.write(full_output)

# Return summary via notebook exit
lines = full_output.strip().split("\n")
summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "no summary")
try:
    dbutils.notebook.exit(f"rc={result.returncode}|{summary}")
except NameError:
    print(f"rc={result.returncode}|{summary}")
